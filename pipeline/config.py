
import sys
import argparse

try:
    from model import EmotionNET, spike_rate_loss
    SNN_AVAILABLE = True
except ImportError:
    SNN_AVAILABLE = False


CONFIG = {
    'dreamer_csv':          './data/EEG_clean_table.csv',
    'seediv_csv':           './data/EEG_all_sessions_combined.csv',
    'out_dir':              './output',

    'fs':                   128,
    'n_channels':           14,
    'window_size':          384,
    'stride':               384,

    'artifact_ptp_uv':      8.0,    # z-score units (DREAMER is MATLAB z-scored)
    'artifact_flat_uv':     0.02,   # z-score std (flat signal threshold)

    'bands': {'delta':(1,4),'theta':(4,8),'alpha':(8,13),'beta':(13,30),'gamma':(30,45)},

    'in_channels':          9,
    'out_channels':         30,
    'lstm_hidden':          64,
    'lstm_layers':          1,
    'dropout':              0.25,
    'n_classes':            4,

    'lr':                   2e-4,
    'weight_decay':         1e-4,
    'lambda_mmd':           0.05,
    'lambda_seed':          1.0,
    'num_epochs':           60,
    'early_stop_patience':  15,
    'warmup_epochs':        10,   # train DREAMER only before CORAL alignment kicks in
    'coral_update_interval': 5,  # recompute CORAL matrix every N epochs
    'batch':                32,
    'val_ratio':            0.15,
    'seed':                 42,

    'subjects':             None,   # None = all

    'snn_threshold'     : 1.0,
    'snn_spike_weight'  : 1e-3,   # spike rate regularization weight
    'snn_target_rate'   : 0.1,    # target mean spike rate
}


def parse_args():
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # ── Subject / mode ────────────────────────────────────────────────────
        parser.add_argument('--subjects',     type=int, nargs='+', default=None,
                            help='Subject IDs to run (default: all)')
        parser.add_argument('--smoke',        action='store_true',
                            help='Quick smoke test: 1 subject, 3 epochs, 50 segs')
        parser.add_argument('--snn',          action='store_true',
                            help='Use SNN model (requires snntorch)')
        parser.add_argument('--dreamer-only', action='store_true',
                            help='Benchmark: DREAMER LOSO only, no SEED-IV')
        parser.add_argument('--seediv-only',  action='store_true',
                            help='Benchmark: SEED-IV LOSO only, no DREAMER')
        parser.add_argument('--no-coral',     action='store_true',
                            help='Disable latent CORAL alignment')

        # ── Loss weights ──────────────────────────────────────────────────────
        parser.add_argument('--lambda-seed',  type=float, default=None,
                            help='Weight for SEED-IV CE loss (default: CONFIG value)')
        parser.add_argument('--lambda-mmd',   type=float, default=None,
                            help='Weight for MMD loss (default: CONFIG value)')

        # ── Training ──────────────────────────────────────────────────────────
        parser.add_argument('--epochs',       type=int,   default=None,
                            help='Number of training epochs')
        parser.add_argument('--batch',        type=int,   default=None,
                            help='Batch size')
        parser.add_argument('--lr',           type=float, default=None,
                            help='Learning rate')
        parser.add_argument('--patience',     type=int,   default=None,
                            help='Early stopping patience')
        parser.add_argument('--warmup',       type=int,   default=None,
                            help='Warmup epochs (DREAMER only before CORAL kicks in)')
        parser.add_argument('--dropout',      type=float, default=None,
                            help='Dropout rate')
        parser.add_argument('--seed',         type=int,   default=None,
                            help='Random seed')

        # ── Model ─────────────────────────────────────────────────────────────
        parser.add_argument('--out-channels', type=int,   default=None,
                            help='CNN out_channels (ANN) / snn_out_channels (SNN)')
        parser.add_argument('--lstm-hidden',  type=int,   default=None,
                            help='LSTM hidden size')
        parser.add_argument('--window',       type=int,   default=None,
                            help='Window size in samples (default 384 = 3s@128Hz)')

        # ── SNN-specific ──────────────────────────────────────────────────────
        parser.add_argument('--snn-beta1',    type=float, default=None,
                            help='LIF1 beta (membrane decay)')
        parser.add_argument('--snn-beta2',    type=float, default=None,
                            help='LIF2 beta (membrane decay)')
        parser.add_argument('--snn-threshold',type=float, default=None,
                            help='LIF spike threshold')

        # ── Output ────────────────────────────────────────────────────────────
        parser.add_argument('--run-name',     type=str,   default=None,
                            help='Run name tag for output files (e.g. lr1e4_lmbd0.5). '
                                'Output: output/<run-name>/  Default: output/')

        args = parser.parse_args()
        return args



def apply_args_to_config(args, CONFIG):
        if args.subjects:                          CONFIG['subjects']             = args.subjects
        if args.lambda_seed  is not None:          CONFIG['lambda_seed']          = args.lambda_seed
        if args.lambda_mmd   is not None:          CONFIG['lambda_mmd']           = args.lambda_mmd
        if args.epochs       is not None:          CONFIG['num_epochs']           = args.epochs
        if args.batch        is not None:          CONFIG['batch']                = args.batch
        if args.lr           is not None:          CONFIG['lr']                   = args.lr
        if args.patience     is not None:          CONFIG['early_stop_patience']  = args.patience
        if args.warmup       is not None:          CONFIG['warmup_epochs']        = args.warmup
        if args.dropout      is not None:          CONFIG['dropout']              = args.dropout
        if args.seed         is not None:          CONFIG['seed']                 = args.seed
        if args.out_channels is not None:          CONFIG['out_channels']         = args.out_channels
        if args.lstm_hidden  is not None:          CONFIG['lstm_hidden']          = args.lstm_hidden
        if args.window       is not None:          CONFIG['window_size']          = args.window
        if args.snn_beta1    is not None:          CONFIG['snn_beta1']            = args.snn_beta1
        if args.snn_beta2    is not None:          CONFIG['snn_beta2']            = args.snn_beta2
        if args.snn_threshold is not None:         CONFIG['snn_threshold']        = args.snn_threshold
        if args.lstm_hidden  is not None:          CONFIG['snn_lstm_hidden']      = args.lstm_hidden
        if args.run_name     is not None:
            CONFIG['out_dir']    = f'./output/{args.run_name}'
            CONFIG['_run_name']  = args.run_name

        if args.smoke:
            print('\n⚡ SMOKE TEST MODE: 1 subject, 3 epochs, 50 segs cap\n')
            CONFIG['subjects']            = [1]
            CONFIG['num_epochs']          = 3
            CONFIG['early_stop_patience'] = 99
            CONFIG['_smoke']              = True
        else:
            CONFIG['_smoke'] = False

        if args.snn:
            if not SNN_AVAILABLE:
                print('ERROR: snntorch not installed. Cannot use --snn flag.')
                sys.exit(1)
            CONFIG['snn_out_channels'] = 50
            CONFIG['snn_hidden']       = 50
            CONFIG['snn_beta1']        = 0.9
            CONFIG['snn_beta2']        = 0.9
            CONFIG['snn_threshold']    = 0.3
            CONFIG['snn_spike_weight'] = 0.0
            CONFIG['snn_target_rate']  = 0.1
            CONFIG['_use_snn'] = True
            print('\n🧠 SNN MODE: CNN + Delta Encoder + 2 Dense\n')
        else:
            CONFIG['_use_snn'] = False

        # Benchmark modes
        CONFIG['_dreamer_only'] = args.dreamer_only
        CONFIG['_seediv_only']  = args.seediv_only
        CONFIG['_no_coral']     = args.no_coral
        if args.no_coral:
            print('\n⚠️  CORAL disabled — CE_S uses raw SEED-IV latent\n')
        if args.dreamer_only and args.seediv_only:
            print('ERROR: cannot use --dreamer-only and --seediv-only together.')
            sys.exit(1)
        if args.dreamer_only:
            print('\n📊 BENCHMARK MODE: DREAMER LOSO only (no SEED-IV)\n')
        if args.seediv_only:
            print('\n📊 BENCHMARK MODE: SEED-IV LOSO only (no DREAMER)\n')
        
        return CONFIG

 