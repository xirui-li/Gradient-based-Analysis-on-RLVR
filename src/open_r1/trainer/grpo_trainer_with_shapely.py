from .grpo_trainer import GRPOTrainer

class GRPOTrainerWithShapely(GRPOTrainer):
    """
    GRPOTrainer with Shapely Value for enhanced gradient-based analysis.
    
    This class extends the GRPOTrainer to include Shapely functionalities,
    Apart from the train set training calculation, this class will also do the same pipeline to evaluation set.
    The shapely value is obtained by the dot product of the weight vector from train set and the evaluation set.
    Since the evaluation set is not used for training, it will not be used to update the model weights.
    And the reward functions for evaluaiton set will only keep the accuracy reward function.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Additional initialization for Shapely can be added here if needed

        # Validation reward functions, weights, processing classes only keep those for accuracy reward
        if self.reward_functions is not None:
            self.reward_functions_evaluation = {
                "accuracy": self.reward_functions["accuracy"],
            }
        if self.reward_weights is not None:
            self.reward_weights_evaluation = {
                "accuracy": args.reward_weights["accuracy"],
            }
        if self.processing_classes is not None:
            self.processing_classes_evalution = {
                "accuracy": self.processing_classes["accuracy"],
            }

    def get_eval_dataloaders(self, *args, **kwargs):
        """
        Override to include Shapely-specific evaluation data loaders if needed.
        """
        if self.eval_dataset is None:
            raise ValueError("Trainer: Shapely Value requires a eval_dataset.")

        eval_dataset = self.eval_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        dataloader_params = {
            "batch_size": self._eval_batch_size * self.args.steps_per_generation,  # < this is the change
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))

    def _get_eval_sampler(self, dataset: Optional[Dataset] = None) -> Sampler:
        # Returns a sampler that
        # 1. ensures each prompt is repeated across multiple processes. This guarantees that identical prompts are
        #    distributed to different GPUs, allowing rewards to be computed and normalized correctly within each prompt
        #    group. Using the same seed across processes ensures consistent prompt assignment, preventing discrepancies
        #    in group formation.
        # 2. repeats the batch multiple times to allow reusing generations across multiple updates. Refer to
        #    _prepare_inputs to see how the generations are stored and reused.

        # In the following figure, the values are the prompt indices. The first row shows the first sampled batch, the
        # second row shows the second sampled batch, and so on.
        #
        #                                      |   GPU 0  |   GPU 1  |
        #
        #                 global_step   step    <-───>  num_generations=2
        #                                       <-───────> per_device_train_batch_size=3
        #  grad_accum    ▲  ▲  0          0     0   0   1   1   2   2   <- Generate for the first `steps_per_generation` (prompts 0 to 11); store the completions; use the first slice to compute the loss
        #     =2         ▼  |  0          1     3   3   4   4   5   5   <- Take the stored generations and use the second slice to compute the loss
        #                   |
        #                   |  1          2     6   6   7   7   8   8   <- Take the stored generations and use the third slice to compute the loss
        #  steps_per_gen=4  ▼  1          3     9   9  10  10  11  11   <- Take the stored generations and use the fourth slice to compute the loss
        #
        #                      2          4    12  12  13  13  14  14   <- Generate for the second `steps_per_generation` (prompts 12 to 23); store the completions; use the first slice to compute the loss
        #                      2          5    15  15  16  16  17  17   <- Take the stored generations and use the second slice to compute the loss
        #                                          ...
        if dataset is None:
            dataset = self.eval_dataset
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def train_with_shapely(self, model_path: Optional[str] = None, trial: Union["optuna.Trial", Dict[str, Any]] = None):
        """
        Main training entry point. This method will initialize the model, optimizer, and scheduler,
        set up the training loop, and handle the training process. At each step, two inputs will be passed to the model:
        1. The training set input, which will be used to update the model weights. Each training step will have a new training batch.
        2. The evaluation set input, which will be used to compute the Shapely value. Each training step will take the evaluation set as as batch and 
            use the same evaluation set input, which is the same for all training steps.

        Args:
            model_path (:obj:`str`, `optional`):
                Local path to the model if the model to train has been instantiated from a local path. If present,
                training will resume from the optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
        """
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)

        # Model re-init
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed(self.args.seed)

            model = self.call_model_init(trial)

            self.model = model.to(self.args.device)

            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Keeping track whether we can can len() on the dataset or not
        train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)
        eval_dataset_is_sized = isinstance(self.eval_dataset, collections.abc.Sized)

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        eval_dataloader = self.get_eval_dataloader() # This is the evaluation set for Shapely Value

        # This is the evaluation batch for Shapely Value
        eval_batch = None
        if eval_dataloader is not None:
            # Get a single batch from evaluation dataset and reuse it
            eval_iterator = iter(eval_dataloader)
            eval_batch = next(eval_iterator)
            # Move to appropriate device
            if isinstance(eval_batch, dict):
                eval_batch = {k: v.to(self.args.device) for k, v in eval_batch.items()}
            else:
                eval_batch = eval_batch.to(self.args.device)

        # Check if evaluation dataset is bigger than a training batch
        if eval_dataloader is not None and eval_dataset_is_sized and len(eval_dataloader) > self.args.train_batch_size:
            raise ValueError(
                "The evaluation dataset is bigger than the training batch size. "
                "Please increase the training batch size or reduce the evaluation dataset size."
            )

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        if train_dataset_is_sized:
            num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if self.args.max_steps > 0:
                max_steps = self.args.max_steps
                num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                    self.args.max_steps % num_update_steps_per_epoch > 0
                )
            else:
                max_steps = math.ceil(self.args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(self.args.num_train_epochs)
        else:
            # see __init__. max_steps is set when the dataset has no __len__
            max_steps = self.args.max_steps
            num_train_epochs = 1
            num_update_steps_per_epoch = max_steps

        self.create_optimizer_and_scheduler(num_training_steps=max_steps)
        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(model_path)

        # Mixed precision training with apex (torch < 1.6)
        model = self.model
        if self.args.fp16 and _use_apex:
            if not is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, self.optimizer = amp.initialize(model, self.optimizer, opt_level=self.args.fp16_opt_level)

        # Multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=(
                    not getattr(model.config, "gradient_checkpointing", False)
                    if isinstance(model, PreTrainedModel)
                    else True
                ),
            )
        # find_unused_parameters breaks checkpointing as per
        # https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021

        # Train!
        if is_torch_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
            )

        num_examples = (
            self.num_examples(train_dataloader)
            if train_dataset_is_sized
            else total_train_batch_size * self.args.max_steps
        )

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", num_examples)
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", max_steps)

        self.state.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        # Check if continuing training from a checkpoint
        if model_path and os.path.isfile(os.path.join(model_path, "trainer_state.json")):
            self.state = TrainerState.load_from_json(os.path.join(model_path, "trainer_state.json"))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", self.state.global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
        self.state.trial_params = hp_params(trial) if trial is not None else None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        tr_loss = torch.tensor(0.0).to(self.args.device)
        self._logging_loss_scalar = 0
        self._globalstep_last_logged = 0
        self._total_flos = self.state.total_flos
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(self.args, self.state, self.control)

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
                    self.args.device
                )
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            steps_in_epoch = len(epoch_iterator) if train_dataset_is_sized else self.args.max_steps
            self.control = self.callback_handler.on_epoch_begin(self.args, self.state, self.control)

            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(self.args, self.state, self.control)

                if (
                    ((step + 1) % self.args.gradient_accumulation_steps != 0)
                    and self.args.local_rank != -1
                    and _use_ddp_no_sync
                ):
                    with model.no_sync():
                        tr_loss += self.training_step(model, inputs)
                else:
                    tr_loss += self.training_step(model, inputs)
                self._total_flos += self.floating_point_ops(inputs)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= self.args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    if self.args.fp16 and _use_native_amp:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                    elif self.args.fp16 and _use_apex:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    if is_torch_tpu_available():
                        xm.optimizer_step(self.optimizer)
                    elif self.args.fp16 and _use_native_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.lr_scheduler.step()
                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(self.args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            self.control = self.callback_handler.on_epoch_end(self.args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch)

            if self.args.tpu_metrics_debug or self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if self.args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )
            if isinstance(model, PreTrainedModel):
                self.model = model.from_pretrained(self.state.best_model_checkpoint)
                self.model = self.model.to(self.args.device)
            else:
                state_dict = torch.load(os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME))
                self.model.load_state_dict(state_dict)

        if self._total_flos is not None:
            self.store_flos()
            self.log({"total_flos": self.state.total_flos})

        self.control = self.callback_handler.on_train_end(self.args, self.state, self.control)

        return TrainOutput(self.state.global_step, tr_loss.item() / self.state.global_step)

    def training_step(self, model: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
        """
        Full training step with gradient stats collection (Accelerate-compatible) on both train set and evaluation set.
        """
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        if is_sagemaker_mp_enabled():
            print("is_sagemaker_mp_enabled")
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        del inputs

        if self.args.torch_empty_cache_steps is not None and self.state.global_step % self.args.torch_empty_cache_steps == 0:
            torch.cuda.empty_cache()

        kwargs = {}

        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                loss = loss / self.args.gradient_accumulation_steps

            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False

            self.accelerator.backward(loss, **kwargs)
            gradients = self._extract_global_gradients(self.accelerator, self.model)

            return loss.detach()

    def training_step_with_shapley(self, model: nn.Module, inputs: Dict[str, Any], eval_inputs: Dict[str, Any]) -> torch.Tensor:
        """
        Full training step with Shapley value computation using gradient dot products.
        Computes gradients on both training and evaluation batches for Shapley calculation.
        """
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        eval_inputs = self._prepare_inputs(eval_inputs)
        
        if is_sagemaker_mp_enabled():
            print("is_sagemaker_mp_enabled")
            # Note: SageMaker MP path would need custom handling for dual gradient computation
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        # Compute loss and gradients for training batch
        with self.compute_loss_context_manager():
            train_loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
        
        # Compute loss and gradients for evaluation batch
        with self.compute_loss_context_manager():
            eval_loss = self.compute_loss(model, eval_inputs, num_items_in_batch=num_items_in_batch)

        # Clean up inputs
        del inputs
        del eval_inputs

        if self.args.torch_empty_cache_steps is not None and self.state.global_step % self.args.torch_empty_cache_steps == 0:
            torch.cuda.empty_cache()

        kwargs = {}
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            train_loss = train_loss.mean()
            eval_loss = eval_loss.mean()

        # Compute gradients for training loss
        if self.use_apex:
            with amp.scale_loss(train_loss, self.optimizer) as scaled_loss:
                train_gradients = torch.autograd.grad(
                    scaled_loss, 
                    model.parameters(), 
                    retain_graph=True, 
                    create_graph=True
                )
        else:
            if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                train_loss_scaled = train_loss / self.args.gradient_accumulation_steps
            else:
                train_loss_scaled = train_loss
                
            train_gradients = torch.autograd.grad(
                train_loss_scaled,
                model.parameters(),
                retain_graph=True,
                create_graph=Fasle # First order gradients are sufficient for Shapley computation
            )

        # Compute gradients for evaluation loss
        if self.use_apex:
            with amp.scale_loss(eval_loss, self.optimizer) as scaled_loss:
                eval_gradients = torch.autograd.grad(
                    scaled_loss,
                    model.parameters(),
                    retain_graph=True,
                    create_graph=False  # First order gradients are sufficient for Shapley computation
                )
        else:
            if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                eval_loss_scaled = eval_loss / self.args.gradient_accumulation_steps
            else:
                eval_loss_scaled = eval_loss
                
            eval_gradients = torch.autograd.grad(
                eval_loss_scaled,
                model.parameters(),
                retain_graph=True,
                create_graph=False
            )

        # Compute Shapley value as dot product of gradients
        shapley_value = self.compute_shapley_from_gradients(train_gradients, eval_gradients)
        
        # Log Shapley value for monitoring
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({"shapley_value": shapley_value.item()})

        # Now perform the actual backward pass for training
        if self.use_apex:
            with amp.scale_loss(train_loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                final_loss = train_loss / self.args.gradient_accumulation_steps
            else:
                final_loss = train_loss

            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False

            self.accelerator.backward(final_loss, **kwargs)

        # Extract gradients for any additional processing
        gradients = self._extract_global_gradients(self.accelerator, self.model)

        return train_loss.detach()

    def compute_shapley_from_gradients(self, train_gradients, eval_gradients) -> torch.Tensor:
        """
        Compute Shapley value as dot product of training and evaluation gradients.
        
        Args:
            train_gradients: Tuple of gradient tensors from training batch
            eval_gradients: Tuple of gradient tensors from evaluation batch
        
        Returns:
            Shapley value as scalar tensor
        """
        shapley_value = torch.tensor(0.0, device=self.args.device)
        
        # Compute dot product for each parameter's gradients
        for train_grad, eval_grad in zip(train_gradients, eval_gradients):
            if train_grad is not None and eval_grad is not None:
                # Flatten gradients and compute dot product
                train_flat = train_grad.flatten()
                eval_flat = eval_grad.flatten()
                shapley_value += torch.dot(train_flat, eval_flat)
        
        return shapley_value