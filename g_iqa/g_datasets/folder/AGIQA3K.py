from g_iqa.g_datasets.folder.PIQ23_base import PIQ23Folder


class AGIQA3KFolder(PIQ23Folder):
    def __init__(self, root, data_json, transform):
        super().__init__(root, data_json, transform)