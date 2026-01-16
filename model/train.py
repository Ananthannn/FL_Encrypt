def train_model(x, y, model, optimizer, loss, metrics, epochs, batch_size):
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=list(metrics)
    )

    model.fit(
        x, y,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    return model.get_weights()
