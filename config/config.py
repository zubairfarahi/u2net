import configparser
import os

def load_configuration():
    config_file = os.getenv('CONFIG_FILE', 'config.ini')

    config = configparser.ConfigParser()
    config.read(config_file)

    print(f"Using config file: {config_file}")

    try:
        # Model Training Parameters
        epoch_num = config.getint('model_training', 'epoch_num')
        batch_size_train = config.getint('model_training', 'batch_size_train')
        eps = config.getfloat('model_training', 'eps')
        weight_decay = config.getfloat('model_training', 'weight_decay')

        # Optimizer Parameters
        lr = config.getfloat('optimizer', 'lr')
        betas = eval(config.get('optimizer', 'betas'), {"__builtins__": None}, {})

        # Paths for Data, Model Saving, and Debugging
        data_train_path = config.get('paths', 'data_training_path')
        data_val_path = config.get('paths', 'data_validation_path')
        model_dir = config.get('paths', 'save_models_path')
        model_destination = config.get('paths', 'save_top_models_path')
        debug_path = config.get('paths', 'debug_path')
        pretrained_model_path = config.get('paths', 'pretrained_model_path')
        input_path = config.get('testing', 'input')
        output_path = config.get('testing', 'output')

        # Model Settings
        model_type = config.get('model_settings', 'model_type')
        debug_mode = config.getboolean('model_settings', 'debug_mode')
        visualize_filters = config.getboolean("model_settings", "visualize_filters")
        testing_flag = config.getboolean("testing", "flag")
        distributed = config.getboolean("model_settings", "distributed")
        # CUDA Configuration
        device = config.getint('cuda', 'gpu')

    except configparser.NoOptionError as e:
        print(f"Error: Missing required configuration option - {e}")
        raise
    except ValueError as e:
        print(f"Error: Invalid configuration value - {e}")
        raise

    # Return a dictionary of the loaded configuration
    return {
        "epoch_num": epoch_num,
        "batch_size_train": batch_size_train,
        "eps": eps,
        "weight_decay": weight_decay,
        "lr": lr,
        "betas": betas,
        "data_train_path": data_train_path,
        "data_val_path": data_val_path,
        "model_dir": model_dir,
        "model_type": model_type,
        "model_dest": model_destination,
        "cuda": device,
        "debug_mode": debug_mode,
        "debug_path": debug_path,
        "pretrained_model_path": pretrained_model_path,
        "visualize_filters": visualize_filters,
        "testing_flag": testing_flag,
        "input": input_path,
        "output": output_path,
        "distributed": distributed
    }