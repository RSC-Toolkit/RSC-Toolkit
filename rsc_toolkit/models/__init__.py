from rsc_toolkit.models.factory import ModelFactory


ModelFactory.register_models()



def create_dataset(args):
    model_class = ModelFactory.get_model_class()
    return model_class()
