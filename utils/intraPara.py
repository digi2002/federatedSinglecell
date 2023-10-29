#save
datasets ={
    'AMB':['/AMB/Filtered_mouse_allen_brain_data.csv','/AMB/Labels.csv'],
    'BaronHuman':['/Pancreatic_data/Baron Human/Filtered_Baron_HumanPancreas_data.csv','/Pancreatic_data/Baron Human/Labels.csv'],
    'BaronMouse':['/Pancreatic_data/Baron Mouse/Filtered_MousePancreas_data.csv','/Pancreatic_data/Baron Mouse/Labels.csv'],
    'Muraro':['/Pancreatic_data/Muraro/Filtered_Muraro_HumanPancreas_data.csv','/Pancreatic_data/Muraro/Labels.csv'],
    'Segerstolpe':['/Pancreatic_data/Segerstolpe/Filtered_Segerstolpe_HumanPancreas_data.csv','/Pancreatic_data/Segerstolpe/Labels.csv'],
    'Xin':['/Pancreatic_data/Xin/Filtered_Xin_HumanPancreas_data.csv','/Pancreatic_data/Xin/Labels.csv'],
    'TM':['/TM/Filtered_TM_data.csv','/TM/Labels.csv'],
    'zhengsorted':['/Zheng sorted/Filtered_DownSampled_SortedPBMC_data.csv','/Zheng sorted/Labels.csv']
}

def get_Datasets():
    return datasets