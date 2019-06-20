def is_chinese(uchar):
    return u'\u4e00' <= uchar <= u'\u9fa5'


def is_number(uchar):
    return u'\u0030' <= uchar <= u'\u0039'


def is_alphabet(uchar):
    return (u'\u0041' <= uchar <= u'\u005a') or (u'\u0061' <= uchar <= u'\u007a')


def is_valid(uchar):
    return is_chinese(uchar) or is_number(uchar) or is_alphabet(uchar)\
           or uchar == u'\u000a' or uchar == u'\u002d'
