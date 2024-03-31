import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import data_loader.extract_data as extract_data
import data_loader.load_data as load_data
import visualize.visualize as visualize
from fine_tune.finetune import TrainModelClassifier
def run_finetune_convs(medical_type, model_type, batch_size, additional_evaluation = False):
    """
    Performs fine-tuning of a conv models on a given medical type and model type.
    :param medical_type: The type of medical data to classify ('ucsd', 'ori').
    :param model_type: The type of model to use for classification ().
    :param batch_size: The batch size for data loading.
    """
    generators, lengths = load_data.create_loader(medical_type, batch_size,finetune = True)
    visualize.save_random_images_from_generators(generators, [medical_type, model_type, "finetune"], 2)
    if model_type in ["resnet50", "alexnet", "vgg19", "squeezenet", "densenet", "inception", "googlenet", "mobilenet_v3_large", "wide_resnet50_2", "mnasnet"]:
        classifier = TrainModelClassifier(medical_type,model_type)
        classifier.run(generators, lengths)
        if additional_evaluation:
            generators, lengths = load_data.create_loader(additional_evaluation, batch_size,finetune = False)
            steps = {"Train":lengths["Train"], "Validation":lengths["Validation"], "Test":lengths["Test"]}
            acc, prec, rec, auc, cr, cm = classifier.evaluate(generators, steps)
            classifier.save_results(acc, prec, rec, auc, cr, cm, additional_evaluation = True)
        return classifier
    else:
        print("Did not define a proper classifer!")

if __name__ == "__main__":
    extract_data.mount_and_process()
    batch_size = 128
    models_types = ["resnet50", "alexnet", "vgg19", "squeezenet", "densenet", "inception", "googlenet", "mobilenet_v3_large", "wide_resnet50_2", "mnasnet"]
    medical_types = ['ucsd', 'ori']
    for i in range(len(models_types)):
        ucsd_classifier = run_finetune_convs(medical_types[0], models_types[i], batch_size, additional_evaluation=medical_types[1])
        ori_classifier = run_finetune_convs(medical_types[1], models_types[i], batch_size, additional_evaluation=medical_types[0])
  