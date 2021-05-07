import typing

import kfp.components as comp


def confusion_matrix(
    labels_dir: comp.InputPath(),
) -> typing.NamedTuple("conf_m_result", [("mlpipeline_ui_metadata", "UI_metadata")]):
    import json
    from collections import namedtuple

    from sklearn.metrics import confusion_matrix

    # Load class names
    with open(f"{labels_dir}/class_names.txt", "r") as fc:
        class_names = eval(fc.read())

    # Load test labels and predicted labels
    with open(f"{labels_dir}/true_labels.txt", "r") as ft:
        test_labels = eval(ft.read())

    with open(f"{labels_dir}/pred_labels.txt", "r") as fp:
        pred_labels = eval(fp.read())

    # Build confusion matrix
    confusion_matrix = confusion_matrix(test_labels, pred_labels)

    csv_literal_confusion_matrix = ""
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            csv_literal_confusion_matrix += "{target},{predicted},{count}\n".format(
                target=class_names[i],
                predicted=class_names[j],
                count=confusion_matrix[i][j],
            )

    kf_literal_confusion_matrix = {
        "outputs": [
            {
                "type": "confusion_matrix",
                "format": "csv",
                "schema": [
                    {"name": "target", "type": "CATEGORY"},
                    {"name": "predicted", "type": "CATEGORY"},
                    {"name": "count", "type": "NUMBER"},
                ],
                "storage": "inline",
                "source": csv_literal_confusion_matrix,
                "labels": class_names,
            }
        ]
    }

    confusion_matrix_result = namedtuple("conf_m_result", ["mlpipeline_ui_metadata"])
    return confusion_matrix_result(json.dumps(kf_literal_confusion_matrix))
