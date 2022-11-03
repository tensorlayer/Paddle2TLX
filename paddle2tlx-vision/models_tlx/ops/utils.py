def channels_switching(data_format, dim='2d', padding=None):
    if dim == '1d':
        if data_format == 'channels_first':
            out = 'NCL'
        if data_format == 'channels_last':
            out = 'NLC'
        pads = padding
    if dim == '2d':
        if data_format == 'channels_first':
            out = 'NCHW'
        if data_format == 'channels_last':
            out = 'NHWC'
        pads = [padding[1][0], padding[1][1], padding[0][0], padding[0][1]]
    if dim == '3d':
        if data_format == 'channels_first':
            out = 'NCDHW'
        if data_format == 'channels_last':
            out = 'NDHWC'
    # return out
        pads = [padding[2][0], padding[2][1],
                padding[1][0], padding[1][1],
                padding[0][0], padding[0][1]]
    return out, pads

