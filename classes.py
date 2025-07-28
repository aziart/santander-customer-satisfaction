class Paths:
    """
    A class to hold paths for the project.
    """
    RAW_DATA_DIR = "../data/raw"
    PROCESSED_DATA_DIR = "../data/processed"
    MODELS_DIR = "../models"
    LOGS_DIR = "../logs"
    NOTEBOOKS_DIR = "../notebooks"
    SRC_DIR = "../src"

    @classmethod
    def get_raw_data_path(cls, filename: str) -> str:
        return f"{cls.RAW_DATA_DIR}/{filename}"

    @classmethod
    def get_processed_data_path(cls, filename: str) -> str:
        return f"{cls.PROCESSED_DATA_DIR}/{filename}"

    @classmethod
    def get_model_path(cls, filename: str) -> str:
        return f"{cls.MODELS_DIR}/{filename}"

    @classmethod
    def get_log_path(cls, filename: str) -> str:
        return f"{cls.LOGS_DIR}/{filename}"

    @classmethod
    def get_notebook_path(cls, filename: str) -> str:
        return f"{cls.NOTEBOOKS_DIR}/{filename}"

    @classmethod
    def get_src_path(cls, filename: str) -> str:
        return f"{cls.SRC_DIR}/{filename}"