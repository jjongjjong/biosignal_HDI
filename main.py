from excute import excute
from config import config_list

if __name__=='__main__':
    
    configs = config_list()[:]
    for config_instance in configs:
        excute(config_instance)
