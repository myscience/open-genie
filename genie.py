from lightning.pytorch.cli import LightningCLI

from genie import Genie
from genie.dataset import LightningPlatformer2D

def cli_main():
    '''
    Main function for the training script.
    '''
    
    # That's all it takes for LightningCLI to work!
    # No need to call .fit() or .test() or anything like that.
    cli = LightningCLI(
        Genie,
        LightningPlatformer2D,
    )

if __name__ == '__main__':    
    cli_main()