import os
import io
import time
import torch
import queue
import pstats
import cProfile
import numpy as np
import scipy.stats.mstats

from threading import Thread
from multiprocessing import Process
from diagmodels.models import modelbase as dptmodelbase
from digitalpathology.utils import serialize as dptserialize
from diagmodels.models.pytorch.pytorchmodelbase import PytorchModelBase


class DummyModel(PytorchModelBase):
    def build(self):
        return

    def calculate_loss(self):
        return

    def calculate_metric(self):
        return


class async_tile_processor(Process):

    def __init__(self, read_queue, write_queues, model_path, augment=False, soft=False, batch_size=16,
                 ax_order='whc', preprocess_batch=None, gpu_device=-1, unfix_network=False, tile_size=None):
        Process.__init__(self, name='TileProcessor')

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)

        self.daemon = True
        self._model = None

        self._model_path = model_path
        self._read_queue = read_queue
        self._write_queues = write_queues
        self._augment = augment
        self._soft = soft
        self._fast_read_queue = None
        self._tile_size = tile_size
        self._postprocess_function = None
        self._max_batch = batch_size
        self._preprocess_batch = preprocess_batch
        self._ax_order = ax_order
        self._unfix_network = unfix_network

    def _load_network_model(self):
        """
        Load network model and instantiate it.
        Args:
            model_path (str): Path of the model file to load.
            required_shape (tuple): Processing patch shape.
            unrestrict_network (bool): Fix network for fixed in and output sizes.
        Returns:
            digitalpathology.models.modelbase.ModelBase: Loaded network instance.
        Raises:
            MissingModelFileError: Missing model file.
            DigitalPathologyModelError: Model errors.
        """

        # Check if the model path is a valid file path.
        # Load the network data structure separately, in case the input shape needs to be unfixed.
        network_data = dptserialize.load_object(path=self._model_path)

        # Return the prepared model.
        network = dptmodelbase.ModelBase.instantiate(file=network_data)

        if self._unfix_network:
            network.unfix((self._tile_size, self._tile_size))

        return network

    def _predict_tile_batch(self, tile_batch=None, info=None):
        """
        Performs classification of an image in one go.
        This function using the prediction functions to classify the image.
        If image is not given to this function, it will fill this variable
        from self._inputs. Optionally, msk_data can be provided to mask
        out certain parts of the image.
        """
        if self._preprocess_batch:
            self._preprocess_batch(tile_batch, info)

        if self._ax_order == 'cwh':
            tile_batch = tile_batch.transpose((0, 3, 1, 2))
        result = self._model.predict(tile_batch)['predictions']
        if isinstance(result, list):
            result = np.stack([x[0] for x in result])
        if self._ax_order == 'cwh':
            result = result.transpose((0, 2, 3, 1))

        return result

    def _process_tile_batch(self, tile_batch=None, info=None):
        """
        Processes the tile batch on the GPU. Optionally performs test time augmentation.
        Args:
            tile_batch (np.array):
            info (tuple):
        Returns:
        """
        if self._augment:
            result_stack = list()
            for flip in (True, False):
                flipped_tile = np.flip(m=tile_batch, axis=1) if flip else tile_batch
                for rot_k in range(4):
                    augmented_tile = np.rot90(m=flipped_tile, k=rot_k, axes=(1, 2))
                    augmented_result = self._predict_tile_batch(tile_batch=augmented_tile, info=info)
                    augmented_result = np.rot90(m=augmented_result, k=-rot_k, axes=(1, 2))
                    augmented_result = np.flip(m=augmented_result, axis=1) if flip else augmented_result
                    result_stack.append(augmented_result)

            return scipy.stats.mstats.gmean(a=np.stack(result_stack), axis=0)
        else:
            return self._predict_tile_batch(tile_batch=tile_batch, info=info)

    def _stealer_daemon(self, source_queue, dest_queue):
        """
        Starts the thread for transferring data from the Process.queue to the Thread.queue.
        Args:
            source_queue (Process.queue): Queue that is being filled with batch data from the reader.
            dest_queue (Thread.queue): This queue will be read from during the
        """

        def steal(source_queue, dest_queue):
            """
            This function is must be started in a thread.
            Args:
                source_queue (Process.queue): Source.
                dest_queue (Thread.queue): Destination.
            """
            while True:
                obj = source_queue.get()
                dest_queue.put(obj)

        stealer = Thread(target=steal, args=(source_queue, dest_queue))
        stealer.daemon = True
        stealer.start()

    def _run_loop(self):
        # pr = cProfile.Profile()
        # pr.enable()
        while True:
            tile_info = self._fast_read_queue.get()
            writer_nr = tile_info[-1]
            if tile_info[0] == 'finish_image':
                self._write_queues[writer_nr].put(tile_info[:-1])
                continue
            output_filename, sequence_nr, tile_batch, mask_batch, info, _ = tile_info
            result_batch = self._process_tile_batch(tile_batch, info)
            self._write_queues[writer_nr].put(('write_tile', output_filename, sequence_nr, result_batch, mask_batch, info))
            # if np.random.rand() < 0.1: self._reset_and_dump_profiler(pr)

    def _send_reconstruction_info(self):
        in_shape = (self._tile_size, self._tile_size, 3)
        if self._ax_order == 'cwh':
            in_shape = (3, self._tile_size, self._tile_size)
        recon_info = self._model.getreconstructioninformation(in_shape)
        self._write_queues[0].put((
            'recon_info',
            [recon_info[0].astype(np.int32),
             recon_info[1].astype(np.int32),
             recon_info[2].astype(np.int32)],
            self._model.getnumberofoutputchannels()))

    def _initiate_fast_queue(self):
        """
        Initiates the threaded queue which is used to transfer and store the data from the processed queue.
        """
        self._fast_read_queue = queue.Queue(maxsize=10)
        self._stealer_daemon(self._read_queue, self._fast_read_queue)

    def _reset_and_dump_profiler(self, profiler, sort_mode='cumulative'):
        profiler.disable()
        s = io.StringIO()
        sort_by = sort_mode
        ps = pstats.Stats(profiler, stream=s).sort_stats(sort_by)
        ps.print_stats()
        print(s.getvalue())
        profiler.enable()

    def run(self):
        """
        Called when the process is started. This runs the main steps of the tile processor.
        """
        self._model = self._load_network_model()
        self._send_reconstruction_info()
        self._initiate_fast_queue()
        self._run_loop()


class vanilla_tile_processor(async_tile_processor):
    def _load_pytorch_model(self, pytorch_model, checkpoint):
        self._model_object = pytorch_model
        self._model_object.load_state_dict(torch.load(checkpoint))
        self._model_object.eval()

    def _load_network_model(self):
        dpt_model = DummyModel()
        assert self._model_object
        model = self._model_object
        model.cuda()
        model.eval()
        dpt_model._model_instance = model

        return dpt_model

