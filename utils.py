from skimage.transform import resize
import gym
import torch

def get_screen(env, device, h=80, w=80):
	"""
	Returns a screen render of the environment as torch Tensor.

	Parameters:
		env : gym environment object
		h : height of render
		w : width of render
		device : store location of the render

	Returns:
		k : tensor with screen render
	"""

	k = env.render(mode='rgb_array')
	k = resize(k, (h, w), anti_aliasing=False)
	k = torch.Tensor(k)
	k = k.permute(2,1,0).unsqueeze(0)
	k = k.to(device)
	return k

def evaluate(no_seeds=10):
	"""Function to evaluate trained models using greedy actions.
	"""
	r_vec = []
	for i in range(no_seeds):
		env_test.seed(i)
		env_test.reset()
		Rt = 0
		for timesteps in count():
			# choose greedy action
			action = select_action(tnet(get_screen(env_test)), wnet.head.weight.data.view(-1,1), greedy=True)
			_, R, done, _ = env_test.step(action.item())
			Rt = R + Rt
			if(done):
				r_vec.append(R)
				break
				
	return np.mean(r_vec), np.std(r_vec) 