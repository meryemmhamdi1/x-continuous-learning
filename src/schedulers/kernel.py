import numpy as np

class Kernel(object):
    def __init__(self,
                 kern,
                 nu,
                 val_accs,
                 val_losses,
                 train_accs,
                 train_losses,
                 delays,
                 curr_batch
                 ):
        """
            loss_i: the loss of the network for a training instance h_i
            delay_i: the number of epochs to the next review 
            acc_e: the performance of the network on the validation data 
        """
        self.kern = kern
        self.nu = nu
        self.val_accs = val_accs
        self.val_losses = val_losses
        self.val_strength = np.mean(val_accs)
        self.train_accs = train_accs
        self.train_losses = train_losses
        self.delays = delays
        self.curr_batch = curr_batch

    def get_optimal_tau_rbf(self):
        x = 1.0 / self.val_strength
        if self.kern == 'gau':
            a_ln = -1. * np.sum([np.log(a) for a in self.val_accs if a >= self.nu])
            x_sum_pow = np.sum([pow(l * x, 2) for l, a in zip(self.val_losses, self.val_accs) if a >= self.nu])
            tau = a_ln / x_sum_pow
        
        if self.kern == 'lap':
            a_ln = -1. * np.sum([np.log(a) for a in self.val_accs if a >= self.nu])
            x_sum = np.sum([l * x for l, a in zip(self.val_losses, self.val_accs) if a >= self.nu])                        
            tau = a_ln / x_sum
        
        if self.kern == 'lin':
            a_one = np.sum([(1. - a) for a in self.val_accs if a >= self.nu])
            x_sum = np.sum([l * x for l, a in zip(self.val_losses, self.val_accs) if a >= self.nu])                            
            tau = a_one / x_sum
        
        if self.kern == 'cos':
            a_arc = np.sum([np.arccos(2. * a - 1.) for a in self.val_accs if a >= self.nu])
            x_sum = np.sum([l * x for l, a in zip(self.val_losses, self.val_accs) if a >= self.nu])
            tau = a_arc / (np.pi * x_sum)
        
        if self.kern == 'qua':
            a_one = np.sum([(1. - a) for a in self.val_accs if a >= self.nu])
            x_sum_pow = np.sum([pow(l * x, 2) for l, a in zip(self.val_losses, self.val_accs) if a >= self.nu])           
            tau = a_one / x_sum_pow
        
        if self.kern == 'sec':
            a_sq = np.sum([np.log(1. / a + np.sqrt(1. / a - 1.)) for a in self.val_accs if a >= self.nu])
            x_sum = np.sum([l * x for l, a in zip(self.val_losses, self.val_accs) if a >= self.nu])                            
            tau = a_sq / x_sum
        return tau 

    def get_optimal_delay(self):
        print("len(self.train_accs):", len(self.train_accs), " len(self.curr_batch):", len(self.curr_batch))
        tau = self.get_optimal_tau_rbf()
        self.train_accs = np.array([a if a < 1. else .9 for a in self.train_accs]) 
        if self.kern == 'gau':
            nu_gau = np.sqrt(-np.log(self.nu) / tau)
            for i in range(len(self.curr_batch)):
                if self.train_accs[i] >= self.nu:
                    self.delays[self.curr_batch[i]] = max(1., self.val_strength * nu_gau / self.train_losses[i])
        
        if self.kern == 'lap':
            nu_lap = np.log(self.nu)    
            for i in range(len(self.curr_batch)):
                if self.train_accs[i] >= self.nu:
                    self.delays[self.curr_batch[i]] = max(1., -1. * self.val_strength * nu_lap / (self.train_losses[i] * tau))
        
        if self.kern == 'lin':
            nu_lin = (1. - self.nu)
            for i in range(len(self.curr_batch)):
                if self.train_accs[i] >= self.nu:
                    self.delays[self.curr_batch[i]] = max(1., self.val_strength * nu_lin / (self.train_losses[i] * tau))
        
        if self.kern == 'cos':
            nu_cos = np.arccos(2 * self.nu - 1.)
            for i in range(len(self.curr_batch)):
                if self.train_accs[i] >= self.nu:
                    self.delays[self.curr_batch[i]] = max(1., self.val_strength * nu_cos / (np.pi * self.train_losses[i] * tau))
        
        if self.kern == 'qua':
            nu_qua = np.sqrt((1. - self.nu) / tau)
            for i in range(len(self.curr_batch)):
                if self.train_accs[i] >= self.nu:
                    self.delays[self.curr_batch[i]] = max(1., self.val_strength * nu_qua / self.train_losses[i])
            
        if self.kern == 'sec':
            nu_sec = np.log(1. / self.nu * (1 + np.sqrt(1 - self.nu * self.nu)))
            for i in range(len(self.curr_batch)):
                if self.train_accs[i] >= self.nu:
                    self.delays[self.curr_batch[i]] = max(1., self.val_strength * nu_sec / (self.train_losses[i] * tau))

        return self.delays

