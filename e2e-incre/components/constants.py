PAD_TOKEN = '<blank>'
BOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
UNK_TOKEN = '<unk>'
NAME_TOKEN = '<name>'
NEAR_TOKEN = '<near>'

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

START_VOCAB = [PAD_TOKEN,
               BOS_TOKEN,
               EOS_TOKEN,
               UNK_TOKEN,
               ]

MR_FIELDS = ["name", "familyFriendly", "eatType", "food", "priceRange", "near", "area", "customer rating"]
MR_KEYMAP = dict((key, idx) for idx, key in enumerate(MR_FIELDS))
