
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from matplotlib.widgets import Button,Slider, TextBox, RadioButtons

import re
from src.functions import *
from matplotlib import rc
import textwrap
import random
import json
from src.config import *
import datetime
import sys
from src.config import *
# rc('text', usetex=True)

class User_study_interface():

    def __init__(self, data_ditributions, samples_per_data=1, interaction_samples=1, dis_names=None, model=None, dev_mode=False, mr=None):
        """data_ditributions: list, with the different data distributions
        samples_per_data: int number of samples per distribution to present the user
        dis_names: list of with the distributuions names"""

        self.dev_mode = dev_mode
        self.model = model
        self.dis_names = [str(i) for i in range(len(data_ditributions))] if dis_names is None else dis_names
        self.num_samples = samples_per_data*len(data_ditributions) # per ditribution
        exp_data = []
        exp_data_indices = []
        exp_sample_indices = []

        self.interaction_text = {}
        self.samples_inices = {}
        self.user_answers = {}
        self.interaction_samples = interaction_samples

        self.mr = mr
        if interaction_samples > 0 and mr is None:
            print("For interactive mode, please provide the motion refiner mr object")
            return

        for i,data_dis in enumerate(data_ditributions):

            samples_inices = random.choices(range(len(data_dis)), k=samples_per_data)
            # print(self.dis_names[i],samples_inices)
            exp_data = exp_data + np.array(data_dis)[samples_inices].tolist()
            
            # exp_data = exp_data + random.choices(data_dis, k=samples_per_data)
            self.samples_inices[self.dis_names[i]] = samples_inices
            
            exp_data_indices = exp_data_indices + [i]*samples_per_data
            exp_sample_indices = exp_sample_indices + samples_inices


        index_shuf = list(range(len(exp_data)))
        random.shuffle(index_shuf)
        # print(index_shuf)
        self.exp_data = [exp_data[i].copy() for i in index_shuf]
        self.exp_data_indices = [exp_data_indices[i] for i in index_shuf]
        self.exp_sample_indices = [exp_sample_indices[i] for i in index_shuf]


        data_dis = data_ditributions[0]
        samples_inices = random.choices(range(len(data_dis)), k=interaction_samples)
        self.exp_data  = self.exp_data + np.array(data_dis)[samples_inices].tolist()
        self.samples_inices["interaction"] = samples_inices
        self.exp_data_indices = self.exp_data_indices + [i]*interaction_samples
        self.exp_sample_indices = self.exp_sample_indices + samples_inices

        self.w = textwrap.TextWrapper(width=60,break_long_words=False,replace_whitespace=False)
        
        self.init_setup()

    def init_setup(self):
        self.fig = plt.figure(figsize=(8,13))
        self.fig.add_subplot(1,1,1,projection='3d')
        self.ax = plt.gca(projection='3d')
        # self.ax.set_xlabel('X axis')
        # self.ax.set_ylabel('Y axis')
        # self.ax.set_zlabel('Z axis')
        if not self.dev_mode:
            self.ax.axes.xaxis.set_ticklabels([])
            self.ax.axes.yaxis.set_ticklabels([])
            self.ax.axes.zaxis.set_ticklabels([])

        self.ani=None
        self.lines = []
        self.tip_marker = []
        self.objs_scatter = []
        self.objs_text = []
        self.colors = []
        self.enable_save = False
        self.user_data = {"name":"", "age":""}
        self.eval_text_list = ['yes, much better', 'yes, a bit better' , 'same', 'No, a bit wrong', 'No, totally wrong']

        self.sample_i =1

        self.alert_text = None
        self.count_text = self.ax.text2D(0.05, 0.95, str(self.sample_i)+"/"+str(self.num_samples+self.interaction_samples), transform=self.ax.transAxes)


        self.setup_bts()
        if self.sample_i > self.num_samples:
            if self.sample_i <= self.num_samples+self.interaction_samples:
                self.reset()
                self.plot_intercative()
            else:
                pass
        else:
            self.plot_sample(self.exp_data[self.sample_i-1])

            self.reset_bts()

        self.plot_markers()

        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press_event)
        plt.show()

    def update(self, num, trajs, lines, tips):

        for i, line in enumerate(lines):
            line.set_data(trajs[i][:2,:num])
            line.set_3d_properties(trajs[i][2,:num])
        for i, tip in enumerate(tips):
            n = max(num-1, 0)
            tip.set_data(trajs[i][:2,n:n+1])
            tip.set_3d_properties(trajs[i][2,n:n+1])
    

    def on_key_press_event(self, event):
        sys.stdout.flush()
        if event.key == 'enter':
            if self.sample_i > self.num_samples:
                if self.sample_i <= self.num_samples+self.interaction_samples:
                    self.predict(event)

    def next_cb(self, event):
        if self.sample_i in self.user_answers.keys():
            self.sample_i += 1
            if self.sample_i > self.num_samples:
                if self.sample_i <= self.num_samples+self.interaction_samples:
                    self.reset()
                    self.plot_intercative()
                else:
                    self.reset()
                    self.plot_final_screen()
            
            else:
                self.plot_sample(self.exp_data[self.sample_i-1])
                self.reset_bts()

        else:
            if not self.alert_text is None:
                self.alert_text.remove()
            self.alert_text = self.ax.text2D(0.50, -0.1,"Please answer the question first!", transform=self.ax.transAxes, color="red")
        

    def prev_cb(self, event):
        if self.sample_i > 1:
            self.sample_i -= 1
            if self.sample_i > self.num_samples:
                if self.sample_i <= self.num_samples+self.interaction_samples:
                    self.reset()
                    self.plot_intercative()
                else:
                    pass
            else:
                self.plot_sample(self.exp_data[self.sample_i-1])

                self.reset_bts()


    def eval_cb(self, labels):
        self.user_answers[self.sample_i] = self.eval_text_list.index(labels)

        # self.radio.remove()
    

    def plot_final_screen(self):
        self.final_bg = plt.axes([0.05, 0.05, 0.9, 0.9], facecolor="white")


        self.final_bg.set_xticks([])
        self.final_bg.set_yticks([])
        final_text = self.final_bg.text(0.05, 0.7,"Thank you for helping us :)", transform=self.final_bg.transAxes, color="Black", fontweight="bold", fontsize=27)

        self.save_text = self.final_bg.text(0.2, 0.63," ", transform=self.final_bg.transAxes, color="white")

        self.ax_name_box = plt.axes([0.2, 0.5, 0.4, 0.04])
        self.name_box = TextBox(self.ax_name_box, 'your name:', initial=self.user_data["name"])
        self.name_box.on_submit(lambda val: self.log_entry(val,"name"))

        self.ax_age_box = plt.axes([0.2, 0.4, 0.4, 0.04])
        self.age_box = TextBox(self.ax_age_box, 'your age:', initial=self.user_data["age"])
        self.age_box.on_submit(lambda val: self.log_entry(val,"age"))

        self.axsave = plt.axes([0.2, 0.2, 0.20, 0.075])
        self.btsave = Button(self.axsave, 'SAVE and EXIT')
        self.btsave.on_clicked(self.save_data)
    
    def plot_intercative(self):
        # self.interactive_bg = plt.axes([0.05, 0.7, 0.9, 0.3], facecolor="white")
        # self.interactive_bg.set_xticks([])
        # self.interactive_bg.set_yticks([])
        
        # final_text = self.final_bg.text(0.05, 0.7,"Thank you for helping us :)", transform=self.final_bg.transAxes, color="Black", fontweight="bold", fontsize=27)

        # self.save_text = self.final_bg.text(0.2, 0.63," ", transform=self.final_bg.transAxes, color="white")

        self.nl_box_ax = plt.axes([0.2, 0.85, 0.4, 0.04])
        init_text = "" if not self.sample_i in self.interaction_text.keys() else self.interaction_text[self.sample_i]
        self.nl_box = TextBox(self.nl_box_ax, '', initial=init_text )
        self.nl_box.on_submit(self.register_interaction)

        self.axmodify = plt.axes([0.6, 0.85, 0.20, 0.040])
        self.btmodify = Button(self.axmodify, 'modify trajectory')
        self.btmodify.on_clicked(self.predict)


        if self.dev_mode:
            self.lf_ax = plt.axes([0.2, 0.80, 0.4, 0.04])
            self.lf_slider = Slider(
                ax=self.lf_ax,
                label='Locality factor',
                valmin=0.0,
                valmax=1.0,
                valinit=self.exp_data[self.sample_i-1]["locality_factor"],
            )
            self.lf_slider.on_changed(self.update_lf)


            # register the update function with each slider


        plot_out = self.sample_i in self.interaction_text.keys()
        self.plot_sample(self.exp_data[self.sample_i-1],plot_out=plot_out)
        self.ax.set_title(" Type your instruction:\n\n\n\n\n\n", fontsize=18,  fontname="Times New Roman")
        self.reset_bts()

    def update_lf(self, val):
        self.exp_data[self.sample_i-1]["locality_factor"] = val

    def register_interaction(self, text):
        self.interaction_text[self.sample_i] = text

    def predict(self, event):
        if self.sample_i in self.interaction_text.keys():
            d = self.exp_data[self.sample_i-1]
            pred, traj = self.mr.apply_interaction(self.model, d, self.interaction_text[self.sample_i],  label=False, images=None)
            self.pred=pred[0]
            self.exp_data[self.sample_i-1]["output_traj"] = pred[0].tolist()
            # x_t = (mr.prepare_x(X_test), list_to_wp_seq(y_test,d=4), X_test[:,embedding_indices])
            # self.pred = generate(model ,x_t, traj_n=traj_n).numpy()
            print(self.pred.shape)
            self.plot_sample(self.exp_data[self.sample_i-1],new_pred=True)
            self.ax.set_title(" Type your instruction:\n\n\n\n\n", fontsize=18,  fontname="Times New Roman")
            self.reset_bts()
        else:
            print("write first")

    def log_entry(self, val, k):
        self.user_data[k] = str(val)

        self.enable_save = True
        for key in self.user_data.keys():
            if self.user_data[key] == "":
                self.enable_save = False

    def save_data(self, name):
        if self.enable_save:
            if not os.path.exists(user_study_folder):
                print("User study folder not found:", user_study_folder)

            summary = {d_name:{a:0 for a in self.eval_text_list} for d_name in self.dis_names}
            answers_per_distribution = {d_name:{a:[] for a in self.eval_text_list} for d_name in self.dis_names}

            for i,(k,v) in enumerate(self.user_answers.items()):
                print("V",v, "  K: ",k)
                dis = self.dis_names[self.exp_data_indices[int(k-1)]]
                answer = self.eval_text_list[v]
                summary[dis][answer]= summary[dis][answer] + 1
                print(self.exp_sample_indices[i],"\t")
                
                answers_per_distribution[dis][answer] = answers_per_distribution[dis][answer] + [self.exp_sample_indices[i]]

            data_to_save = {"summary":summary,
                            "answers_per_distribution":answers_per_distribution,
                            "num_samples":self.num_samples,
                            "dis_names":self.dis_names,
                            "exp_data":self.exp_data,
                            "exp_data_indices":self.exp_data_indices,
                            "exp_sample_indices":self.exp_sample_indices,
                            "user_answers":self.user_answers,
                            "interaction_text":self.interaction_text
                            }
            # data_to_save
            print(self.user_answers)
            with open( os.path.join(user_study_folder, self.user_data["name"]+"_"+self.user_data["age"] +"_"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S") +'.json'), 'w') as f:
                json.dump(data_to_save, f)
            plt.close()
        else: 
            if not self.save_text is None:
                self.save_text.remove()
            self.save_text = self.final_bg.text(0.2, 0.63,"Please complete all the fields first", transform=self.final_bg.transAxes, color="red")
    
    def replay_from_file(self,user_name):
        print(user_name)
        u = {}
        with open(os.path.join(user_study_folder,user_name), 'r', encoding='utf-8') as f:
            u = json.load(f)

        self.num_samples = u["num_samples"]
        self.dis_names = u["dis_names"]
        self.exp_data = u["exp_data"]
        self.exp_data_indices = u["exp_data_indices"]
        self.exp_sample_indices = u["exp_sample_indices"]
        # self.interaction_text = "interaction_text"
        self.sample_i =1

        for k,v in u["interaction_text"].items():
            self.interaction_text[int(k)]=v
        for k,v in u["user_answers"].items():
            self.user_answers[int(k)]=int(v)
        self.init_setup()
        # self.user_answers = u["user_answers"]

        # for k,v in u.items():
        #     print(k,v)
        print(self.user_answers)
        print(self.sample_i)

        if self.sample_i in self.user_answers.keys() or str(self.sample_i) in self.user_answers.keys():
            print("A")
            print(self.user_answers[self.sample_i])
            print(type(self.user_answers[self.sample_i]))




    def setup_bts(self):
        
        self.alert_text = self.ax.text2D(0.70, 0.05, "alert", transform=self.ax.transAxes, color="white")
        axcolor = 'lightgoldenrodyellow'

        self.rax = plt.axes([0.15, 0.05, 0.2, 0.15],
                    facecolor=axcolor)
                    
        self.rax.set_title("Does the modified trajectory \n follows the intruction?")

        self.axnext = plt.axes([0.66, 0.05, 0.15, 0.075])
        if self.sample_i < self.num_samples+self.interaction_samples:
            self.bnext = Button(self.axnext, 'Next')
        elif self.sample_i == self.num_samples+self.interaction_samples:
            self.bnext = Button(self.axnext, 'FINISH!')
        

        self.bnext.on_clicked(self.next_cb)

        self.axprev = plt.axes([0.5, 0.05, 0.15, 0.075])
        if self.sample_i > 1:
            self.bprev = Button(self.axprev, 'Previous')

        else:
            self.bprev = Button(self.axprev, 'Previous',color="white")

        self.bprev.on_clicked(self.prev_cb)

        # adjust radio buttons
        self.radio = RadioButtons(self.rax, self.eval_text_list,
                            active=[True,False,False,False],
                            activecolor='r')

        self.radio.on_clicked(self.eval_cb)
        if self.sample_i in self.user_answers.keys() or str(self.sample_i) in self.user_answers.keys():
            print("A")
            self.radio.set_active(self.user_answers[self.sample_i])


        
    def reset_bts(self):
        self.axprev.remove()
        self.axnext.remove()
        self.rax.remove()
        self.setup_bts()
        
    def plot_sample(self, d, new_pred=False, plot_out=True, no_title=False):
        
        self.reset()
        
        pts = np.asarray(d["input_traj"])
        if new_pred:
            pts_new = self.pred
        else:
            pts_new = np.asarray(d["output_traj"])
        
        text = d["text"]
        obj_names = np.asarray(d["obj_names"])
        obj_pt = np.asarray(d["obj_poses"])
        # change_type = d["change_type"]
        image_paths = d["image_paths"]

        objs  = {}
        for x,y,z,name in zip(obj_pt[:,0],obj_pt[:,1],obj_pt[:,2],obj_names):
            objs[name] = {"value":{"obj_p":[x,y,z]}}

        new_pts_list = [pts_new]
        if d["output_traj"] is None:
            new_pts_list = []

        N = 100
        dt = 0.02
        traj_original = incorporate_speed(pts,dt=dt, N=N)
        traj_new = incorporate_speed(pts_new,dt=dt, N=N)
        # fig = plot_samples(text,pts,new_pts_list, objs=objs,fig=figure, show= False, plot_speed=False, labels=["modified"])
        # if not plot_out:
        #     traj_new = traj_original.copy()
        self.plot_objs(objs)
        if plot_out:
            self.trajs = [traj_original.T, traj_new.T]
        else:
            self.trajs = [traj_original.T] 

        self.plot_trajs(traj_original, traj_new,plot_out=plot_out)

        if not no_title:
            self.ax.set_title("INSTRUCTION:\n\n"+  '\n'.join(self.w.wrap(text))  , fontsize=18,  fontname="Times New Roman")
        else:

            self.ax.set_title(" ", fontsize=18,  fontname="Times New Roman")
        set_axes_equal(self.ax)

        self.animate(N=100,plot_out=plot_out)

    def plot_objs(self, objs):
        for i,(name,v) in enumerate(objs.items()):
            x,y,z = v["value"]["obj_p"]
            
            color = self.colors[i] if i < len(self.colors)-1 else "#"+''.join([random.choice('0123456789AB') for j in range(6)])    

            sc = self.ax.scatter(x,y,z, color=color, s=50)
            # print(dir(sc))
            self.objs_scatter.append(sc)
            t = self.ax.text(x, y, z, name, 'x', color=color, ha='center', fontweight="bold")
            self.objs_text.append(t)

    def plot_trajs(self, pts, pts_new, plot_out=True):

        alpha = 0.1
        color_original = "red"
        color_modified = "blue"

        x_init, y_init, z_init = pts[:,0],pts[:,1],pts[:,2]
        x_new, y_new, z_new= pts_new[:,0],pts_new[:,1],pts_new[:,2]
        

        line3, = self.ax.plot(x_init, y_init, z_init,alpha=0.9,color=color_original, label="Original")
        self.lines.append(line3)
        if plot_out:
            line4, = self.ax.plot(x_new, y_new, z_new,alpha=0.9, color=color_modified, label="Modified")
            self.lines.append(line4)
        handles, labels = self.ax.get_legend_handles_labels()
        self.ax.legend(handles[::-1], labels[::-1])

        line1, = self.ax.plot(x_init, y_init, z_init,alpha=alpha,color=color_original, label="Original")
        self.lines.append(line1)
        if plot_out:
            line2, = self.ax.plot(x_new, y_new, z_new,alpha=alpha, color=color_modified, label="Modified")
            self.lines.append(line2)

        tip_marker1, = self.ax.plot(x_init[0:1], y_init[0:1], z_init[0:1], lw=2, c=color_original, marker='o')
        self.tip_marker.append(tip_marker1)
        if plot_out:
            tip_marker2, = self.ax.plot(x_new[0:1], y_new[0:1], z_new[0:1], lw=2, c=color_modified, marker='o')
            self.tip_marker.append(tip_marker2)

    def plot_markers(self):

        fs=30
        dist = 0.5
        alpha = 0.2
        color = 'grey'
        self.ax.text(0,dist, 0, "front", 'x',color=color, alpha=alpha, fontsize=fs, ha='center', va='center')
        self.ax.text(-dist,0, 0, "left", 'y',color=color, alpha=alpha, fontsize=fs, ha='center', va='center')
        self.ax.text(0,0, -dist, "bottom", 'x',color=color, alpha=alpha, fontsize=fs, ha='center', va='center')

        # self.ax.text(0,-dist, 0, "back", 'x',color='red', alpha=alpha, fontsize=fs, ha='center', va='center')
        # self.ax.text(dist,0, 0, "right", 'y',color='red', alpha=alpha, fontsize=fs, ha='center', va='center')
        # self.ax.text(0,0, dist, "up", 'x',color='red', alpha=alpha, fontsize=fs, ha='center', va='center')


    def reset(self):
        self.count_text.remove()
        self.count_text = self.ax.text2D(0.05, 0.95, str(self.sample_i)+"/"+str(self.num_samples+self.interaction_samples), transform=self.ax.transAxes)
        
        try:
            if not self.alert_text is None:
                self.alert_text.remove()
        except:
            pass

        try:
            self.interactive_bg.remove()
        except:
            pass

        for i in range(len(self.lines)):
            self.lines[i].remove()

        for i in range(len(self.tip_marker)):
            self.tip_marker[i].remove()

        for i in range(len(self.objs_text)):
            # print(self.objs_scatter[i])
            # print("\n",self.objs_scatter[i])
            self.objs_scatter[i].remove()
            self.objs_text[i].remove()

        self.objs_scatter = []
        self.objs_text = []
        self.lines = []
        self.tip_marker = []



    def animate(self, N=100,plot_out=True):
        n_lines = 2 if plot_out else 1
        self.ani = animation.FuncAnimation(self.fig, self.update, N, fargs=(self.trajs,self.lines[:n_lines], self.tip_marker[:n_lines]), interval=1000/N,cache_frame_data=False, blit=False)

