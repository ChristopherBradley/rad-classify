from os.path import join, dirname

# Default stuff
src_folder = dirname(__file__)
root_folder = dirname(src_folder)
data_folder = join(root_folder, 'data')

# WOS stuff
wos_folder = join(data_folder, "WebOfScience") 

WOS5736_X = join(wos_folder, "WOS5736", "X.txt")
WOS5736_Y = join(wos_folder, "WOS5736", "Y.txt")

WOS11967_X = join(wos_folder, "WOS11967", "X.txt")
WOS11967_Y = join(wos_folder, "WOS11967", "Y.txt")

WOS46985_X = join(wos_folder, "WOS46985", "X.txt")
WOS46985_Y = join(wos_folder, "WOS46985", "Y.txt")

# Fasttext stuff
fasttext_train = join(data_folder, 'fasttext_train.csv')
pretrained_vectors = join(data_folder, "wiki-news-300d-1M.vec")

# Preprocessing
pickle_keywords = join(data_folder, 'keywords.pickle')

