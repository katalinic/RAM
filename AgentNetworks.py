
import numpy as np
import Util

class GlimpseNet(object):
    
    def __init__(self, size_g=256, size_hgl=128, size_l=2, size_retina=64):
  
        '''
        size_retina = g_w x g_w (int)
        '''
        self.W_retina = np.random.randn(size_hgl, size_retina)/np.sqrt(size_retina+size_hgl)
        self.b_retina = np.random.randn(size_hgl)/np.sqrt(size_hgl)
        
        self.W_location = np.random.randn(size_hgl, size_l)/np.sqrt(size_hgl)
        self.b_location = np.random.randn(size_hgl)/np.sqrt(size_hgl)
        
        self.W_retina_h_to_g = np.random.randn(size_g, size_hgl)/np.sqrt(size_hgl+size_g)
        self.b_retina_h_to_g = np.random.randn(size_g)/np.sqrt(size_g)
        
        self.W_location_h_to_g = np.random.randn(size_g, size_hgl)/np.sqrt(size_hgl+size_g)
        self.b_location_h_to_g = np.random.randn(size_g)/np.sqrt(size_g)
        
        
        '''
        To be reset to 0 after mini-batch update
        '''
        self.grad_W_retina = np.zeros_like(self.W_retina)
        self.grad_b_retina = np.zeros_like(self.b_retina)
        
        self.grad_W_location = np.zeros_like(self.W_location)
        self.grad_b_location = np.zeros_like(self.b_location)
        
        self.grad_W_retina_h_to_g = np.zeros_like(self.W_retina_h_to_g)
        self.grad_b_retina_h_to_g = np.zeros_like(self.b_retina_h_to_g)
        
        self.grad_W_location_h_to_g = np.zeros_like(self.W_location_h_to_g)
        self.grad_b_location_h_to_g = np.zeros_like(self.b_location_h_to_g)
        
        '''
        Momentum terms
        
        '''
        self.momentum_W_retina = np.zeros_like(self.W_retina)
        self.momentum_b_retina = np.zeros_like(self.b_retina)
        
        self.momentum_W_location = np.zeros_like(self.W_location)
        self.momentum_b_location = np.zeros_like(self.b_location)
        
        self.momentum_W_retina_h_to_g = np.zeros_like(self.W_retina_h_to_g)
        self.momentum_b_retina_h_to_g = np.zeros_like(self.b_retina_h_to_g)
        
        self.momentum_W_location_h_to_g = np.zeros_like(self.W_location_h_to_g)
        self.momentum_b_location_h_to_g = np.zeros_like(self.b_location_h_to_g)
        
        '''
        Storage through time
        '''
        
        self.retina_ReLU = []
        self.location_ReLU = []
        self.h_ReLU = []
        self.retina = []
        self.hg = []
        self.hl = []
        self.l = []
        
    @staticmethod
    def retina(M, l, g):

        s = M.shape[0]

        l1 = np.round((s-1)/2*(l+1)).astype(int)+g

        M_padded = np.pad(M,(g,g),mode='constant')

        g /= 2
        g = int(g)
        
        x_left = l1[0]-g	
        x_right = l1[0]+g
        y_top = l1[1]-g
        y_bottom = l1[1]+g

        return M_padded[y_top:y_bottom,x_left:x_right]
        
    def fwd_pass(self, M, l, g):
        
        '''
        M - image
        l - location
        '''
        self.l.append(l)
        
        self.retina_ReLU.append(Util.ReLU())
        self.location_ReLU.append(Util.ReLU())
        self.h_ReLU.append(Util.ReLU())
        
        self.retina.append(GlimpseNet.retina(M, l, g).ravel())
        
        #Retina linear + ReLU
        hg_input = Util.Linear.fwd_pass(self.retina[-1], self.W_retina, self.b_retina)
        self.hg.append(self.retina_ReLU[-1].fwd_pass(hg_input))
        
        #Location linear + ReLU
        hl_input = Util.Linear.fwd_pass(self.l[-1], self.W_location, self.b_location)
        self.hl.append(self.location_ReLU[-1].fwd_pass(hl_input))
        
        #Retina input to g linear
        linear_hg = Util.Linear.fwd_pass(self.hg[-1], self.W_retina_h_to_g, self.b_retina_h_to_g)
        
        #Location input to g linear
        linear_hl = Util.Linear.fwd_pass(self.hl[-1], self.W_location_h_to_g, self.b_location_h_to_g)
        
        #g
        g_input = Util.Add.fwd_pass(linear_hg, linear_hl)
        g = self.h_ReLU[-1].fwd_pass(g_input)
        
        return g
    
    def back_pass(self, dg):
        
        d_g_input = self.h_ReLU[-1].back_pass(dg)
        
        d_linear_hg, d_linear_hl = Util.Add.back_pass(d_g_input)
        
        d_hl, d_W_location_h_to_g, d_b_location_h_to_g =             Util.Linear.back_pass(d_linear_hl, self.hl[-1], self.W_location_h_to_g)
            
        d_hg, d_W_retina_h_to_g, d_b_retina_h_to_g =             Util.Linear.back_pass(d_linear_hg, self.hg[-1], self.W_retina_h_to_g)
        
        d_hl_input = self.location_ReLU[-1].back_pass(d_hl)
        _, d_W_location, d_b_location = Util.Linear.back_pass(d_hl_input, self.l[-1], self.W_location)
        
        d_hg_input = self.retina_ReLU[-1].back_pass(d_hg)
        _, d_W_retina, d_b_retina = Util.Linear.back_pass(d_hg_input, self.retina[-1], self.W_retina)
        
        '''
        Delete stored instances
        
        '''
        self.retina_ReLU = self.retina_ReLU[:-1]
        self.location_ReLU = self.location_ReLU[:-1]
        self.h_ReLU = self.h_ReLU[:-1]
        self.retina = self.retina[:-1]
        self.hg = self.hg[:-1]
        self.hl = self.hl[:-1]
        self.l = self.l[:-1]
        
        '''
        store the weight deltas
        '''
        
        self.grad_W_retina += d_W_retina
        self.grad_b_retina += d_b_retina
        
        self.grad_W_location += d_W_location
        self.grad_b_location += d_b_location
        
        self.grad_W_retina_h_to_g += d_W_retina_h_to_g
        self.grad_b_retina_h_to_g += d_b_retina_h_to_g
        
        self.grad_W_location_h_to_g += d_W_location_h_to_g
        self.grad_b_location_h_to_g += d_b_location_h_to_g
        
        return None
    
    def reset_storage(self):
        '''
        To be called before each example
        '''
        self.retina_ReLU = []
        self.location_ReLU = []
        self.h_ReLU = []
        self.retina = []
        self.hg = []
        self.hl = []
        self.l = []
        
        return None
    
    def weight_update(self, momentum, learning_rate, batch_size=20):
        
        self.momentum_W_retina *= momentum
        self.momentum_b_retina *= momentum
        
        self.momentum_W_location *= momentum
        self.momentum_b_location *= momentum
        
        self.momentum_W_retina_h_to_g *= momentum
        self.momentum_b_retina_h_to_g *= momentum
        
        self.momentum_W_location_h_to_g *= momentum
        self.momentum_b_location_h_to_g *= momentum
        
        self.momentum_W_retina += learning_rate*self.grad_W_retina/batch_size
        self.momentum_b_retina += learning_rate*self.grad_b_retina/batch_size
        
        self.momentum_W_location += learning_rate*self.grad_W_location/batch_size
        self.momentum_b_location += learning_rate*self.grad_b_location/batch_size
        
        self.momentum_W_retina_h_to_g += learning_rate*self.grad_W_retina_h_to_g/batch_size
        self.momentum_b_retina_h_to_g += learning_rate*self.grad_b_retina_h_to_g/batch_size
        
        self.momentum_W_location_h_to_g += learning_rate*self.grad_W_location_h_to_g/batch_size
        self.momentum_b_location_h_to_g += learning_rate*self.grad_b_location_h_to_g/batch_size
        
        self.W_retina -= self.momentum_W_retina
        self.b_retina -= self.momentum_b_retina
        
        self.W_location -= self.momentum_W_location
        self.b_location -= self.momentum_b_location
        
        self.W_retina_h_to_g -= self.momentum_W_retina_h_to_g
        self.b_retina_h_to_g -= self.momentum_b_retina_h_to_g
        
        self.W_location_h_to_g -= self.momentum_W_location_h_to_g
        self.b_location_h_to_g -= self.momentum_b_location_h_to_g
        
        self.grad_W_retina = np.zeros_like(self.W_retina)
        self.grad_b_retina = np.zeros_like(self.b_retina)
        
        self.grad_W_location = np.zeros_like(self.W_location)
        self.grad_b_location = np.zeros_like(self.b_location)
        
        self.grad_W_retina_h_to_g = np.zeros_like(self.W_retina_h_to_g)
        self.grad_b_retina_h_to_g = np.zeros_like(self.b_retina_h_to_g)
        
        self.grad_W_location_h_to_g = np.zeros_like(self.W_location_h_to_g)
        self.grad_b_location_h_to_g = np.zeros_like(self.b_location_h_to_g)

        return None


class ActionNet(object):
    
    def __init__(self, size_out=10, size_h=256):
        
        self.W_out = np.random.randn(size_out, size_h)/np.sqrt(size_h+size_out)
        self.b_out = np.random.randn(size_out)/np.sqrt(size_out)
        
        self.grad_W_out = np.zeros_like(self.W_out)
        self.grad_b_out = np.zeros_like(self.b_out)
        
        self.momentum_W_out = np.zeros_like(self.W_out)
        self.momentum_b_out = np.zeros_like(self.b_out)
        
        self.softmax = []
        
    def fwd_pass(self, h):
        
        self.softmax.append(Util.Softmax())
        
        softmax_input = Util.Linear.fwd_pass(h, self.W_out, self.b_out)

        return self.softmax[-1].fwd_pass(softmax_input)
        
    def back_pass(self, h, d_output):
        
        d_softmax_input = self.softmax[-1].back_pass(d_output)
        
        d_h, d_W_out, d_b_out = Util.Linear.back_pass(d_softmax_input, h, self.W_out)
        
        self.grad_W_out += d_W_out
        self.grad_b_out += d_b_out
        
        self.softmax = self.softmax[:-1]
        
        return d_h
    
    def reset_storage(self):
        
        self.softmax = []
        
        return None
    
    def weight_update(self, momentum, learning_rate, batch_size=20):
        
        self.momentum_W_out *= momentum
        self.momentum_b_out *= momentum
        
        self.momentum_W_out += learning_rate*self.grad_W_out/batch_size
        self.momentum_b_out += learning_rate*self.grad_b_out/batch_size
        
        self.W_out -= self.momentum_W_out
        self.b_out -= self.momentum_b_out
        
        self.grad_W_out = np.zeros_like(self.W_out)
        self.grad_b_out = np.zeros_like(self.b_out)

        return None


class LocationNet(object):
    
    def __init__(self, sigma, size_h=256, size_l=2):
        
        self.sigma = sigma
        
        self.W_l = np.random.randn(size_l, size_h)/np.sqrt(size_h+size_l)
        self.b_l = np.random.randn(size_l)/np.sqrt(size_l)
        
        self.grad_W_l = np.zeros_like(self.W_l)
        self.grad_b_l = np.zeros_like(self.b_l)
        
        self.momentum_W_l = np.zeros_like(self.W_l)
        self.momentum_b_l = np.zeros_like(self.b_l)
        
        self.Tanh = []
        self.l_tanh = []
        self.l_sampled = []
        
    def fwd_pass(self, h):

        self.Tanh.append(Util.Tanh())
        
        l = Util.Linear.fwd_pass(h, self.W_l, self.b_l)
        
        self.l_tanh.append(self.Tanh[-1].fwd_pass(l))
        
        self.l_sampled.append(np.random.normal(self.l_tanh[-1], self.sigma))
        
        return np.tanh(self.l_sampled[-1])
    
    def back_pass(self, h, r, b):
        
        d_l_tanh = (r-b)*(self.l_sampled[-1]-self.l_tanh[-1])/self.sigma**2
        
        d_l = self.Tanh[-1].back_pass(d_l_tanh)
        
        dh, d_W_l, d_b_l = Util.Linear.back_pass(d_l, h, self.W_l)
        
        self.grad_W_l += d_W_l
        self.grad_b_l += d_b_l
        
        self.Tanh = self.Tanh[:-1]
        self.l_tanh = self.l_tanh[:-1]
        self.l_sampled = self.l_sampled[:-1]
        
        return dh
    
    def reset_storage(self):
        
        self.Tanh = []
        self.l_tanh = []
        self.l_sampled = [] 
        
        return None
    
    def weight_update(self, momentum, learning_rate, batch_size=20):
        
        self.momentum_W_l *= momentum
        self.momentum_b_l *= momentum
        
        self.momentum_W_l += learning_rate*self.grad_W_l/batch_size
        self.momentum_b_l += learning_rate*self.grad_b_l/batch_size
        
        '''
        Want to perform gradient ascent, hence changed - to +
        '''
        self.W_l += self.momentum_W_l
        self.b_l += self.momentum_b_l
        
        self.grad_W_l = np.zeros_like(self.W_l)
        self.grad_b_l = np.zeros_like(self.b_l)
        
        return None


class CoreNet(object):
    
    def __init__(self, size_h=256):
        
        self.W_h_ht = np.random.randn(size_h, size_h)/np.sqrt(size_h+size_h)
        self.b_h_ht = np.random.randn(size_h)/np.sqrt(size_h)
        
        self.grad_W_h_ht = np.zeros_like(self.W_h_ht)
        self.grad_b_h_ht = np.zeros_like(self.b_h_ht)
        
        self.momentum_W_h_ht = np.zeros_like(self.W_h_ht)
        self.momentum_b_h_ht = np.zeros_like(self.b_h_ht)
        
        self.W_h_g = np.random.randn(size_h, size_h)/np.sqrt(size_h+size_h)
        self.b_h_g = np.random.randn(size_h)/np.sqrt(size_h)
        
        self.grad_W_h_g = np.zeros_like(self.W_h_g)
        self.grad_b_h_g = np.zeros_like(self.b_h_g)
        
        self.momentum_W_h_g = np.zeros_like(self.W_h_g)
        self.momentum_b_h_g = np.zeros_like(self.b_h_g)
        
        self.h_ReLU = []
        
    def fwd_pass(self, ht, g):
        
        self.h_ReLU.append(Util.ReLU())
        
        ht_input = Util.Linear.fwd_pass(ht, self.W_h_ht, self.b_h_ht)
        hg_input = Util.Linear.fwd_pass(g, self.W_h_g, self.b_h_g)
        
        h_input = Util.Add.fwd_pass(ht_input, hg_input)
        
        h = self.h_ReLU[-1].fwd_pass(h_input)
        
        return h
    
    def back_pass(self, dh, ht, g):
        
        d_h_input = self.h_ReLU[-1].back_pass(dh)
        
        d_ht_input, d_hg_input = Util.Add.back_pass(d_h_input)
        
        d_g, d_W_h_g, d_b_h_g = Util.Linear.back_pass(d_hg_input, g, self.W_h_g)
        d_ht, d_W_h_ht, d_b_h_ht = Util.Linear.back_pass(d_ht_input, ht, self.W_h_ht)
        
        self.grad_W_h_g += d_W_h_g
        self.grad_b_h_g += d_b_h_g
        
        self.grad_W_h_ht += d_W_h_ht
        self.grad_b_h_ht += d_b_h_ht
        
        self.h_ReLU = self.h_ReLU[:-1]
        
        return d_ht, d_g
    
    def reset_storage(self):
        
        self.h_ReLU = []
        
        return None
    
    def weight_update(self, momentum, learning_rate, batch_size=20):
        
        self.momentum_W_h_ht *= momentum
        self.momentum_b_h_ht *= momentum

        self.momentum_W_h_ht += learning_rate*self.grad_W_h_ht/batch_size
        self.momentum_b_h_ht += learning_rate*self.grad_b_h_ht/batch_size

        self.W_h_ht -= self.momentum_W_h_ht
        self.b_h_ht -= self.momentum_b_h_ht

        self.grad_W_h_ht = np.zeros_like(self.W_h_ht)
        self.grad_b_h_ht = np.zeros_like(self.b_h_ht)
        
        self.momentum_W_h_g *= momentum
        self.momentum_b_h_g *= momentum

        self.momentum_W_h_g += learning_rate*self.grad_W_h_g/batch_size
        self.momentum_b_h_g += learning_rate*self.grad_b_h_g/batch_size

        self.W_h_g -= self.momentum_W_h_g
        self.b_h_g -= self.momentum_b_h_g

        self.grad_W_h_g = np.zeros_like(self.W_h_g)
        self.grad_b_h_g = np.zeros_like(self.b_h_g)
        
        return None

