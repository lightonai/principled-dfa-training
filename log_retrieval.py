# TODO: Select path to log_file.
log_file = ""

bottlenecks = [1, 3, 5, 8, 11, 18, 28, 46, 73, 118, 191, 308, 496, 800]

reading, results = False, []
b = 0

with open(log_file, 'r') as log:
	for line in log.readlines():
		if "EPOCH 50/50" in line:
			reading = True
			results.append({'angles': [], 'acc':[]})
		if reading and 'mean' in line and 'Tanh()' in line:
			angle_data = line.split('mean:')[1]
			angle_data = angle_data.split('std:')
			mean_angle = float(angle_data[0][:-2])
			std_angle = float(angle_data[1][:-1])
			results[-1]['angles'].append([mean_angle, std_angle])
		if reading and 'top-1' in line:
			acc_data = line.split('top-1')
			train_acc = float(acc_data[1][1:5])
			test_acc = float(acc_data[2][1:5])
			results[-1]['acc'] = [train_acc, test_acc]
			print("bottleneck", bottlenecks[b])
			print("train:", train_acc, ", test:", test_acc)
			print(results[-1]['angles'])
			reading = False
			b += 1

print(results)
