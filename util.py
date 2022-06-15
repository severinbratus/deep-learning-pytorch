import torch


def export_model(model, dummy_input, path):
    model.eval()
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(path)
    print(f"Model exported to {path}")


def import_model(path):
    model = torch.jit.load(path)
    model.eval()
    print(f"Model exported from {path}")
    return model
