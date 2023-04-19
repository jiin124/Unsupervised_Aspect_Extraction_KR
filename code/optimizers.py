import tensorflow.keras.optimizers as opt

def get_optimizer(args):

	clipvalue = 0
	clipnorm = 10

	if args.algorithm == 'rmsprop':
		optimizer = opt.RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-06, clipnorm=clipnorm, clipvalue=clipvalue)#lr을 learning_rate=으로 고침
	elif args.algorithm == 'sgd':
		optimizer = opt.SGD(learning_rate=0.01, momentum=0.0, decay=0.0, nesterov=False, clipnorm=clipnorm, clipvalue=clipvalue)
	elif args.algorithm == 'adagrad':
		optimizer = opt.Adagrad(learning_rate=0.01, epsilon=1e-06, clipnorm=clipnorm, clipvalue=clipvalue)
	elif args.algorithm == 'adadelta':
		optimizer = opt.Adadelta(learning_rate=1.0, rho=0.95, epsilon=1e-06, clipnorm=clipnorm, clipvalue=clipvalue)
	elif args.algorithm == 'adam':
		optimizer = opt.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=clipnorm, clipvalue=clipvalue)
	elif args.algorithm == 'adamax':
		optimizer = opt.Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=clipnorm, clipvalue=clipvalue)
	
	return optimizer
