#-*- coding: utf-8 -*-

from module import *
from text.symbols import symbols
import hyperparams as hp
import random

class Encoder(nn.Module):
    """
    Encoder
    """
    def __init__(self, embedding_size):
        """

        :param embedding_size: dimension of embedding
        """
        super(Encoder, self).__init__()
        self.embedding_size = embedding_size
        self.embed = nn.Embedding(len(symbols), embedding_size)
        self.prenet = Prenet(embedding_size, hp.hidden_size * 2, hp.hidden_size)
        self.cbhg = CBHG(hp.hidden_size)

    def forward(self, input_):

        input_ = torch.transpose(self.embed(input_),1,2)
        prenet = self.prenet.forward(input_)
        memory = self.cbhg.forward(prenet)

        return memory

class MelDecoder(nn.Module):
    """
    Decoder
    """
    def __init__(self):
        super(MelDecoder, self).__init__()
        self.prenet = Prenet(hp.num_mels, hp.hidden_size * 2, hp.hidden_size)
        self.attn_decoder = AttentionDecoder(hp.hidden_size * 2)

    def forward(self, decoder_input, memory):

        # Initialize hidden state of GRUcells
        attn_hidden, gru1_hidden, gru2_hidden = self.attn_decoder.inithidden(decoder_input.size()[0])
        outputs = list()

        # Training phase
        if self.training:
            # Prenet
            dec_input = self.prenet.forward(decoder_input)
            timesteps = dec_input.size()[2] // hp.outputs_per_step

            # [GO] Frame
            prev_output = dec_input[:, :, 0]

            for i in range(timesteps):
                prev_output, attn_hidden, gru1_hidden, gru2_hidden = self.attn_decoder.forward(prev_output, memory,
                                                                                             attn_hidden=attn_hidden,
                                                                                             gru1_hidden=gru1_hidden,
                                                                                             gru2_hidden=gru2_hidden)

                outputs.append(prev_output)

                if random.random() < hp.teacher_forcing_ratio:
                    # Get spectrum at rth position
                    prev_output = dec_input[:, :, i * hp.outputs_per_step]
                else:
                    # Get last output
                    prev_output = prev_output[:, :, -1]

            # Concatenate all mel spectrogram
            outputs = torch.cat(outputs, 2)

        else:
            # [GO] Frame
            prev_output = decoder_input

            for i in range(hp.max_iters):
                prev_output = self.prenet.forward(prev_output)
                prev_output = prev_output[:,:,0]
                prev_output, attn_hidden, gru1_hidden, gru2_hidden = self.attn_decoder.forward(prev_output, memory,
                                                                                         attn_hidden=attn_hidden,
                                                                                         gru1_hidden=gru1_hidden,
                                                                                         gru2_hidden=gru2_hidden)
                outputs.append(prev_output)
                prev_output = prev_output[:, :, -1].unsqueeze(2)

            outputs = torch.cat(outputs, 2)

        return outputs

class PostProcessingNet(nn.Module):
    """
    Post-processing Network
    """
    def __init__(self):
        super(PostProcessingNet, self).__init__()
        self.postcbhg = CBHG(hp.hidden_size,
                             K=8,
                             projection_size=hp.num_mels,
                             is_post=True)
        self.linear = SeqLinear(hp.hidden_size * 2,
                                hp.num_freq)

    def forward(self, input_):
        out = self.postcbhg.forward(input_)
        out = self.linear.forward(torch.transpose(out,1,2))

        return out

class Tacotron(nn.Module):
    """
    End-to-end Tacotron Network
    """
    def __init__(self):
        super(Tacotron, self).__init__()
        self.encoder = Encoder(hp.embedding_size)
        self.decoder1 = MelDecoder()
        self.decoder2 = PostProcessingNet()

    def forward(self, characters, mel_input):
        memory = self.encoder.forward(characters)
        mel_output = self.decoder1.forward(mel_input, memory)
        linear_output = self.decoder2.forward(mel_output)

        return mel_output, linear_output