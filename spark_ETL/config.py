class CFG:
    def __init__(self, config_file: str):
        extension = config_file.split(sep='.')[-1]
        if extension == 'json':
            import json
            form = json
            with open(config_file, 'r') as f:
                config = form.load(f)

        elif extension == 'yaml':
            import yaml
            form = yaml
            with open(config_file, 'r') as f:
                config = form.load(f, Loader=yaml.FullLoader)
        else:
            raise TypeError
        for key, value in config.items():
            setattr(self, key, value)
