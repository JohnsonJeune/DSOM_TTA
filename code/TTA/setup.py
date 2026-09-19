import torch.optim as optim
from TTA.adapt_algorithm import norm, oftta, t3a, tent, pl, shot, sar, tast, tast_bn, dsom, cotta, sotta, lame, note, tsd, tea
import torch
import os
import importlib.util

def setup_dsom(args, model):
    model = dsom.configure_model(model)
    tta_model = dsom.dsom(args, model,
                           steps=1,
                           episodic=True)
    return tta_model

# ---- load algorithm modules by file path: avoids module name conflicts, used by standalone implementations such as eata/rotta ----
def _load_alg(filename):
    p = os.path.join(os.path.dirname(__file__), 'adapt_algorithm', filename)
    spec = importlib.util.spec_from_file_location('alg_' + filename.replace('.py', '').replace('+', '_'), p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def setup_source(args, model):
    """Set up the baseline source model without adaptation."""
    model.eval()
    # logger.info(f"model for evaluation: %s", model)
    return model

def setup_norm(args, model):
    """Set up test-time normalization adaptation.

    Adapt by normalizing features with test batch statistics.
    The statistics are measured independently for each batch;
    no running average or other cross-batch estimation is used.
    """
    tta_model = norm.NORM(args, model)
    # logger.info(f"model for adaptation: %s", model)
    stats, stat_names = norm.collect_stats(model)
    # logger.info(f"stats for adaptation: %s", stat_names)
    return tta_model



def setup_pl(args, model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = pl.configure_model(model)
    params, param_names = pl.collect_params(model)
    optimizer = setup_optimizer(args, params)
    tta_model = pl.PL(args, model, optimizer,
                           steps=1,
                           episodic=False)

    return tta_model

def setup_shot(args, model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = shot.configure_model(model)
    params, param_names = shot.collect_params(model)
    optimizer = setup_optimizer(args, params)
    tta_model = shot.SHOT(args, model, optimizer,
                           steps=1,
                           episodic=False)

    return tta_model


def setup_sar(args, model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = sar.configure_model(model)
    params, param_names = sar.collect_params(model)
    base_optimizer = torch.optim.SGD
    optimizer = sar.SAM(params, base_optimizer, lr= args.lr, momentum=0.9)
    tta_model = sar.SAR(args, model, optimizer,
                           steps=1,
                           episodic=False)

    return tta_model

def setup_t3a(args, model):

    model = t3a.configure_model(model)
    params, param_names = t3a.collect_params(model)
    optimizer = setup_optimizer(args, params)
    tta_model = t3a.t3a(args, model, optimizer,
                           steps=1,
                           episodic=False)
    return tta_model

def setup_tast(args, model):

    model = tast.configure_model(model)
    params, param_names = tast.collect_params(model)
    optimizer = setup_optimizer(args, params)
    tta_model = tast.TAST(args, model, optimizer,
                           steps=1,
                           episodic=False)
    return tta_model

def setup_tast_bn(args, model):

    model = tast_bn.configure_model(model)
    params, param_names = tast_bn.collect_params(model)
    optimizer = setup_optimizer(args, params)
    tta_model = tast_bn.TAST_BN(args, model, optimizer,
                           steps=1,
                           episodic=False)
    return tta_model


def setup_oftta(args, model):
    # configure the OFTTA adaptation model: freeze the layers that do not need adaptation (e.g., conv layers) and enable training mode for the layers that do (e.g., BN layers)
    model = oftta.configure_model(model)
    # collect the parameters that need to be optimized (usually BN layer parameters) (usually the weight and bias of BN layers)
    params, param_names = oftta.collect_params(model)
    # initialize the optimizer (actually the Adam optimizer in setup_optimizer)
    optimizer = setup_optimizer(args, params)
    # create the OFTTA adapter instance
    tta_model = oftta.OFTTA(args, model, optimizer,
                           steps=1,       # perform 1 parameter update per batch
                           episodic=False) # disable periodic parameter reset

    return tta_model

def setup_tent(args, model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = setup_optimizer(args, params)
    tta_model = tent.Tent(args, model, optimizer,
                           steps=1,
                           episodic=False)

    return tta_model
def setup_cotta(args, model):
    """Set up CoTTA adaptation model."""
    model = cotta.configure_model(model)
    params, _ = cotta.collect_params(model)
    optimizer = setup_optimizer(args, params)
    tta_model = cotta.CoTTA(args, model, optimizer)
    return tta_model

def setup_note(args, model):
    if args.iabn:
        model = note.convert_iabn(model, args)
    model = note.configure_model(model, args)

    params, _ = note.collect_params(model)
    optimizer = setup_optimizer(args, params)

    note_model = note.NOTE(args, model, optimizer)
    return note_model


def setup_sotta(args, model):
    """
    Initialize the SoTTA pipeline:
    - configure the model (only BN / LN / IN are allowed to update)
    - collect parameters
    - build the optimizer (SAM supported)
    - return the SoTTA instance
    """
    model = sotta.configure_model(model, args)
    params, param_names = sotta.collect_params(model)
    optimizer = setup_optimizer(args, params)

    tta_model = sotta.SoTTA(args, model, optimizer=optimizer,
                      steps=1, episodic=False)

    return tta_model

def setup_lame(args, model):
    """Set up LAME adaptation model.

    Configure the model for evaluation, freeze all parameters,
    initialize affinity and Laplacian label propagation for test-time adaptation.
    """

    model = lame.LAME(args, model)
    return model

def setup_tsd(args, model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = tsd.configure_model(model)
    params, param_names = tsd.collect_params(model)
    optimizer = setup_optimizer(args, params)
    tta_model = tsd.TSD(args, model, optimizer,
                           steps=1,
                           episodic=False)
    return tta_model


def setup_tea(args, model):

    model = tea.configure_model(model)
    params, param_names = tea.collect_params(model)
    optimizer = setup_optimizer(args, params)
    tta_model = tea.TEA(model, optimizer,
                           episodic=False)
    return tta_model

def setup_eata(args, model):
    m = _load_alg('eata_oftta.py')
    return m.EATA(args, model, steps=1, episodic=True)

def setup_rotta(args, model):
    m = _load_alg('rotta_oftta.py')
    return m.RoTTA(args, model, steps=1, episodic=True)

def setup_optimizer(args, params):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if True:
        return optim.Adam(params,
                    lr=args.lr,
                    betas=(0.9, 0.99),
                    weight_decay=5e-4)
    # elif cfg.OPTIM.METHOD == 'SGD':
    #     return optim.SGD(params,
    #                lr=cfg.OPTIM.LR,
    #                momentum=cfg.OPTIM.MOMENTUM,
    #                dampening=cfg.OPTIM.DAMPENING,
    #                weight_decay=cfg.OPTIM.WD,
    #                nesterov=cfg.OPTIM.NESTEROV)
    else:
        raise NotImplementedError

def get_adaptation(args, base_model):
    if args.adaption == 'source':
        print('source')
        model = setup_source(args, base_model)
    elif args.adaption == 'norm':
        print('norm')
        model = setup_norm(args, base_model)
    elif args.adaption == "tent":
        print('tent')
        model = setup_tent(args, base_model)
    elif args.adaption == "t3a":
        print('t3a')
        model = setup_t3a(args, base_model)
    elif args.adaption == "tast":
        print('tast')
        model = setup_tast(args, base_model)
    elif args.adaption == "tast_bn":
        print('tast_bn')
        model = setup_tast_bn(args, base_model)
    elif args.adaption == "oftta":
        print('oftta')
        model = setup_oftta(args, base_model)
    elif args.adaption == "dsom":
        print('dsom')
        model = setup_dsom(args, base_model)
    elif args.adaption == "pl":
        print('pl')
        model = setup_pl(args, base_model)
    elif args.adaption == "shot":
        print('shot')
        model = setup_shot(args, base_model)
    elif args.adaption == "sar":
        print('sar')
        model = setup_sar(args, base_model)
    elif args.adaption == "cotta":
        print('cotta')
        model = setup_cotta(args, base_model)
    elif args.adaption == "sotta":
        print('sotta')
        model = setup_sotta(args, base_model)
    elif args.adaption == "lame":
        print('lame')
        model = setup_lame(args, base_model)
    elif args.adaption == "note":
        print('note')
        model = setup_note(args, base_model)
    elif args.adaption == "tsd":
        print('tsd')
        model = setup_tsd(args, base_model)
    elif args.adaption == "tea":
        print('tea')
        model = setup_tea(args, base_model)
    elif args.adaption == "eata":
        print('eata')
        model = setup_eata(args, base_model)
    elif args.adaption == "rotta":
        print('rotta')
        model = setup_rotta(args, base_model)
    else:
        raise ValueError(
            "not exist this adaptation: %r. Please check whether --adaption is spelled correctly, "
            "or add this branch in get_adaptation of TTA/setup.py." % (args.adaption,))

    return model