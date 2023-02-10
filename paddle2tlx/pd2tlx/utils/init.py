def eval_init_pd(pd_model):
    pd_model.eval()


def eval_init_tlx(tlx_model):
    tlx_model.eval()
    tlx_model.set_eval()
