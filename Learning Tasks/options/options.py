import configargparse

class Options():
    def __init__(self):
        self.parser = configargparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # config file
        self.parser.add_argument('--config', type=str, is_config_file=True, help='config file path')

        # experiment specifics
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--learning_rate', type=float, default=1e-6, help='Learning rate of adam optimizer')
        self.parser.add_argument('--n_epoch', type=int, default=1000, help='Number of epoch for training')
        self.parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
        self.parser.add_argument('--mode', type=str, default='luminancescale', help='What the network learns. luminance, luminancescale, temperature, illuminance, scale, scalebins, scalepretrain, scalebinspretrain.')
        self.parser.add_argument('--early_stop', default=False, action='store_true', help='Whether to stop early based on val_loss (patience 100 epochs)')
        self.parser.add_argument('--existing_in', default=False, action='store_true', help='Use the phase_in image for LDR input')


        # data management
        self.parser.add_argument('--crop', type=int, default=0, help='If to crop the image when prediction illuminance, 0 is not cropped, value is degree of projection, -1 is random between 30-120')
        self.parser.add_argument('--dataroot', type=str, default='./dataset', help='path to images (should have subfolders train, test)')
        self.parser.add_argument('--tonemap_LDR', type=str, default='gamma', help='tonemap method for LDR(log, gamma)')
        self.parser.add_argument('--gamma_LDR', type=float, default=2.4, help='gamma value for gamma tonemap')
        self.parser.add_argument('--clip', default=True, action='store_true', help='Clip the input LDR image to [0,1]')
        self.parser.add_argument('--no_clip', dest='clip', action='store_false', help='dont use the clip the input LDR image to [0,1]')
        self.parser.add_argument('--orig_scale', default=False, action='store_true', help='Rescale the output to original scale after clipping')
        self.parser.add_argument('--quantization', type=int, default=0, help='the quantization of the input image. 0 is None')
        self.parser.add_argument('--color_jitter', type=float, default=0, help='Randomly jitter the color of the input channels. 0 is None')
        self.parser.add_argument('--WBaugmenter', default=False, action='store_true', help='Whether to randomly modify the WB or not')
        self.parser.add_argument('--noise', type=float, default=0, help='Gaussian noise amount on input image. 0 is None')
        self.parser.add_argument('--augmentation', default=False, action='store_true', help='Whether to randomly rotate the azimuth or not')
        self.parser.add_argument('--save_val_img', default=False, action='store_true', help='Whether to save the first validation image at every epoch')


        #metrics
        self.parser.add_argument('--use_solid_angles_map', default=True, action='store_true', help='Use the solid angle map for metrics or not')
        self.parser.add_argument('--no_use_solid_angles_map', dest='use_solid_angles_map', action='store_false', help='dont use the solid angle map for metrics or not')
        self.parser.add_argument('--solid_angles_map', type=str, default='none', help='path to solid angle map')
        self.parser.add_argument('--cos_path', type=str, default='none', help='path to cos map')

        #Network parameters
        self.parser.add_argument('--scale_loss_factor', type=float, default=1e-3, help='multiplication of scale loss to the total loss')
        self.parser.add_argument('--feat', type=int, default=32, help='Number of features at the highest resolution')
        self.parser.add_argument('--down_layers', type=int, default=5, help='Number of downsamplings')
        self.parser.add_argument('--identity_layers', type=int, default=3, help='Number of residual blocks before and after bottleneck. Meaning for a value of 3, we have 6 residual blocks at each level with two convolutions each')
        self.parser.add_argument('--bottleneck_layers', type=int, default=6, help='Number of residuals blocks for bottleneck')
        self.parser.add_argument('--skips', type=bool, default=True, help='Skip connections')
        self.parser.add_argument('--act_fn', type=str, default="relu", help='Activation of inner layers')
        self.parser.add_argument('--out_act_fn', type=str, default="relu", help='Activation after the final layer, usually none')
        self.parser.add_argument('--max_feat', type=int, default=256, help='We doubles features when downsampling but cap it to this value')

        self.parser.add_argument('--script_submodules', default=True, action='store_true', help='Turn on Scripting for faster more efficient network')
        self.parser.add_argument('--no-script_submodules', dest='script_submodules', action='store_false', help='Turn off scripting for faster more efficient network')

        #test
        self.parser.add_argument('--version', type=str, default='None', help='The version of the checkpoint. If None, takes last')

        self.parser.add_argument('--size_x', type=int, default=0, help='Do not touch')
        self.parser.add_argument('--size_y', type=int, default=0, help='Do not touch')



    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_known_args()[0]
        if self.opt.mode == 'luminancescale':
            self.opt.mode = 'luminance' #Legacy support

        return self.opt

    def parseConf(self, arg):
        if not self.initialized:
            self.initialize()
        
        self.opt = self.parser.parse_known_args(['--config', arg])[0]
        if self.opt.mode == 'luminancescale':
            self.opt.mode = 'luminance' #Legacy support

        return self.opt

    def save(self, path, opt):
        self.parser.write_config_file(opt, [path])