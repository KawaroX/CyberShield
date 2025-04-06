def train_model(...):
    # ... existing code ...
    
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=4,  # 将batch_size从16减小到4-8范围
        collate_fn=collate_fn
    )
    
    # ... existing code ...
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        evaluator=evaluator,
        output_path=model_save_path,
        optimizer_params={'lr': learning_rate},
        warmup_steps=100,
        use_amp=True,  # 确保启用混合精度训练
        checkpoint_save_total_limit=1,
        show_progress_bar=True,
        gradient_accumulation_steps=4  # 添加梯度累积
    )
