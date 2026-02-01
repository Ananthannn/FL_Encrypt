import time

def train_model(x, y, model, optimizer, loss, metrics, epochs, batch_size):
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=list(metrics)
    )

    start = time.time()

    history = model.fit(
        x, y,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    end = time.time()
    print(f"⏱ Training time: {end-start:.2f}s")

    return model.get_weights(), history.history

