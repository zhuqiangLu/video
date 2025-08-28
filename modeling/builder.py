class ModelingFuncsBuilder:
    def __init__(self):
        self.model_base_funcs = dict()

    def register(self, model_base, setup_model_func, get_inputs_func, decode_func=None):
        self.model_base_funcs[model_base] = (setup_model_func, get_inputs_func, decode_func)

    def get_modeling_funcs(self, model_base):
        for name, func in self.model_base_funcs.items():
            if name.lower() in model_base.lower():
                return func
        raise ValueError(f"Invalid model base: {model_base}")

modeling_funcs_builder = ModelingFuncsBuilder()
