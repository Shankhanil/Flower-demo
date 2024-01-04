from utils.model import MasterModel
from utils.prepare_dataset import get_mnist
from utils.train import run_centralised



if __name__ == "__main__":

    # Prepare and download the dataset
    # trainset, testset = get_mnist()
    # print(trainset)

    
    # model = MasterModel(num_classes=10)
    # num_parameters = sum(value.numel() for value in model.state_dict().values())
    # print(f"num_parameters = { num_parameters }")
    run_centralised(epochs=5, lr=0.01)