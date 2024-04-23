from resnet_unet import TwoEncodersOneDecoder


class Mandatory2:
    
    def __init__(self):
        self.model = TwoEncodersOneDecoder(
            encoder="resnet18",
        )
    
    def task1(self):
        print("Mandatory2 task1")
        
        

if __name__ == "__main__":
    m = Mandatory2()
    m.task1()
        
        