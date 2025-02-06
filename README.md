# ICAIL 2025

Custom loss functions implemented in PyTorch. They can easily be used with Huggingface Transformers; simply modify the compute_loss() function of the Trainer class:

```python
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    outputs = model(**inputs)
    logits = outputs.get("logits")
    labels = inputs.get("labels")

    if self.loss_type == "IFBCE":
        loss = inverse_freq_weighted_bce(
            labels,
            logits,
            samples_per_cls=self.samples_per_cls or [1] * logits.size(-1),
        no_of_classes=logits.size(-1)
    )
    elif self.loss_type == "FL":
        loss = focal_loss(
            labels,
            logits,
            alpha=self.alpha,
            gamma=self.gamma
        )
    elif self.loss_type == "ASL":
        loss = focal_loss(
            labels,
            logits,
            alpha=self.alpha,
            gamma=self.gamma,
            asymmetric=True
        )
    elif self.loss_type == "CB":
        loss = CB_loss(
            labels,
            logits,
            samples_per_cls=self.samples_per_cls or [1] * logits.size(-1),
            no_of_classes=logits.size(-1),
            loss_type="focal" if self.gamma else "sigmoid",
            beta=self.beta,
            gamma=self.gamma
        )
    elif self.loss_type == "DL":
        loss = dice_loss(
            labels,
            logits
        )
    elif self.loss_type == "TL":
        loss = tversky_loss(
            labels,
            logits,
            alpha=self.alpha
        )
    elif self.loss_type == "FTL":
        loss = focal_tversky_loss(
            labels,
            logits,
            alpha=self.alpha,
            gamma=self.gamma
        )
    elif self.loss_type == "CL":
        loss = combo_loss(
            labels,
            logits,
            alpha=self.alpha,
            lambd=self.lambd
        )
    elif self.loss_type == "AUFL":
        loss = unified_loss(
            labels,
            logits,
            lambd=self.lambd,
            alpha=self.alpha,
            gamma=self.gamma
        )
    elif self.loss_type == "AUFL-CB":
        loss = effective_unified_loss(
            labels,
            logits,
            samples_per_cls=self.samples_per_cls or [1] * logits.size(-1),
            no_of_classes=logits.size(-1),
            lambd=self.lambd,
            beta=self.beta,
            gamma=self.gamma
        )
    else:
        loss = BCEWithLogitsLoss()(logits, labels)

    return (loss, outputs) if return_outputs else loss
```