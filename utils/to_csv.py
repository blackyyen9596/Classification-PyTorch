import pandas as pd


def to_csv(id_read_path, save_path, y_pred):
    my_submission = pd.DataFrame({
        'id': pd.read_csv(id_read_path).id,
        'character': y_pred[:]
    })
    my_submission.to_csv(save_path, index=False)