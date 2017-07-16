import Util
import AgentNetworks

class RAM(object):
    
    def __init__(self,
                glimpse=6,
                hidden_size = 256,
                patch_size = 8,
                momentum = 0.9,
                learning_rate = 1e-3,
                min_learning_rate = 1e-5,
                standard_dev = 0.22,
                batch_size = 20,
                max_epochs = 150):
        
        self.num_glimpses = glimpse
        self.H = hidden_size
        self.g = patch_size
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.sigma = standard_dev
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        
        self.Core = AgentNetworks.CoreNet()
        self.Location = AgentNetworks.LocationNet(self.sigma)
        self.Action = AgentNetworks.ActionNet()
        self.Glimpse = AgentNetworks.GlimpseNet()
        
        self.baseline = 0
        self.grad_baseline = 0
        self.momentum_baseline = 0
        
        self.loss = 0
        self.batch_loss = 0
        self.num_correct = 0
        
        print ("Initialisation complete")
        
    def train_model(self, train_data, target_data):
        
        print ("Commencing training")
        
        for ep in range(self.max_epochs):
            
            for q in range(train_data.shape[0]):
                
                self.Core.reset_storage()
                self.Location.reset_storage()
                self.Action.reset_storage()
                self.Glimpse.reset_storage()

                g_time = np.zeros(self.H)
                h_time = np.zeros(self.H).reshape(1,-1)
                l_time = np.random.uniform(-1,1, size=(1,2))

                #Forward pass
                for i in range(self.num_glimpses):

                    g_time = np.vstack((g_time,self.Glimpse.fwd_pass(train_data[q], l_time[-1], self.g)))

                    h_time = np.vstack((h_time,self.Core.fwd_pass(h_time[-1], g_time[-1])))

                    '''
                    If last step, need to call action, and do not need to call location
                    '''
                    if i==self.num_glimpses-1:

                        output = self.Action.fwd_pass(h_time[-1])

                    else:

                        l_time = np.vstack((l_time, self.Location.fwd_pass(h_time[-1], True)))

                g_time = np.delete(g_time, 0, 0)

                r = 0

                if np.argmax(output)==np.argmax(target_data[q]):

                     r = 1

                #Backprop
                self.grad_baseline += self.baseline-r

                d_output = Util.CrossEntropyLossSoftmax.back_pass(output, target_data[q])

                #Here assuming delta from h_t+1 is 0
                d_h_time = self.Action.back_pass(h_time[-1], d_output)

                d_h, d_g = self.Core.back_pass(d_h_time, h_time[-2], g_time[-1])

                self.Glimpse.back_pass(d_g)
                
                for j in reversed(range(self.num_glimpses-1)):
            
                    d_h_location = self.Location.back_pass(h_time[j+1], r, self.baseline)

                    d_h_time = d_h + d_h_location

                    d_h, d_g = self.Core.back_pass(d_h_time, h_time[j], g_time[j])

                    self.Glimpse.back_pass(d_g)
                    
                #Weight update
                if (q+1)%self.batch_size==0:

                    self.grad_baseline = Util.Clip(self.grad_baseline)
                    self.grad_baseline /= self.batch_size
                    
                    #Perform weight update
                    self.momentum_baseline *= self.momentum
                    self.momentum_baseline += self.learning_rate*self.grad_baseline
                    self.baseline -= self.momentum_baseline
                
                    self.grad_baseline = 0

                    self.Core.weight_update(self.momentum,self.learning_rate)
                    self.Location.weight_update(self.momentum,self.learning_rate)
                    self.Action.weight_update(self.momentum,self.learning_rate)
                    self.Glimpse.weight_update(self.momentum,self.learning_rate)

            self.learning_rate = 0.97*self.learning_rate
            self.learning_rate = max(self.min_learning_rate,self.learning_rate)
            
            
        print ("Training concluded")
        
        return None
    
    def test_model(self, test_data, test_target):

        num_correct = 0

        for q in range(test_data.shape[0]):

            self.Core.reset_storage()
            self.Location.reset_storage()
            self.Action.reset_storage()
            self.Glimpse.reset_storage()

            g_time = np.zeros(self.H)
            h_time = np.zeros(self.H).reshape(1,-1)
            l_time = np.random.uniform(-1,1, size=(1,2))

            #Forward pass
            for i in range(self.num_glimpses):

                g_time = np.vstack((g_time,self.Glimpse.fwd_pass(test_data[q], l_time[-1], self.g)))

                h_time = np.vstack((h_time,self.Core.fwd_pass(h_time[-1], g_time[-1])))

                '''
                If last step, need to call action, and do not need to call location
                '''
                if i==self.num_glimpses-1:

                    output = self.Action.fwd_pass(h_time[-1])

                else:

                    l_time = np.vstack((l_time, self.Location.fwd_pass(h_time[-1], False)))

            r = 0

            if np.argmax(output)==np.argmax(test_target[q]):

                 r = 1

            num_correct += r

        print ("Test classification accuracy: ", num_correct/test_data.shape[0])
        
        return None
