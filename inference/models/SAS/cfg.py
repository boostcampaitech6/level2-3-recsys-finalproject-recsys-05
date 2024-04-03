class CFG:
    def __init__(self, config_file: str = None):
        self._store = {}
        if config_file is not None:
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
                self._store[key] = value
    

    def __getitem__(self, key):
        return self._store[key]
    

    def __setitem__(self, key, value):
        self._store[key] = value


    def __repr__(self):
        return str(self._store)