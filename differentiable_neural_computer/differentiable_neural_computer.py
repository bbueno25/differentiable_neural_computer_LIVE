"""
TODO: docstring
"""
import numpy
import tensorflow

class DifferentiableNeuralComputer:
    """
    3 attention mechanisms for read/writes to memory.
    """
    def __init__(
        self,
        input_size,
        output_size,
        seq_len,
        num_words=256,
        word_size=64,
        num_heads=4):
        """
        TODO: docstring
        """
        self.input_size = input_size
        self.output_size = output_size
        self.num_words = num_words
        self.word_size = word_size
        self.num_heads = num_heads
        self.interface_size = num_heads * word_size + 3 * word_size + 5 * num_heads + 3
        self.nn_input_size = num_heads * word_size + input_size
        self.nn_output_size = output_size + self.interface_size
        self.nn_out = tensorflow.truncated_normal([1, self.output_size], stddev=0.1)
        self.interface_vec = tensorflow.truncated_normal([1, self.interface_size], stddev=0.1)
        self.mem_mat = tensorflow.zeros([num_words, word_size])
        self.usage_vec = tensorflow.fill([num_words, 1], 1e-6)
        self.link_mat = tensorflow.zeros([num_words,num_words])
        self.precedence_weight = tensorflow.zeros([num_words, 1])
        self.read_weights = tensorflow.fill([num_words, num_heads], 1e-6)
        self.write_weights = tensorflow.fill([num_words, 1], 1e-6)
        self.read_vecs = tensorflow.fill([num_heads, word_size], 1e-6)
        self.i_data = tensorflow.placeholder(
            tensorflow.float32, [seq_len*2, self.input_size], name='input_node')
        self.o_data = tensorflow.placeholder(
            tensorflow.float32, [seq_len*2, self.output_size], name='output_node')
        self.W1 = tensorflow.Variable(tensorflow.truncated_normal(
            [self.nn_input_size, 32], stddev=0.1), name='layer1_weights', dtype=tensorflow.float32)
        self.b1 = tensorflow.Variable(tensorflow.zeros([32]), name='layer1_bias', dtype=tensorflow.float32)
        self.W2 = tensorflow.Variable(tensorflow.truncated_normal(
            [32, self.nn_output_size], stddev=0.1), name='layer2_weights', dtype=tensorflow.float32)
        self.b2 = tensorflow.Variable(tensorflow.zeros(
            [self.nn_output_size]), name='layer2_bias', dtype=tensorflow.float32)
        self.nn_out_weights = tensorflow.Variable(tensorflow.truncated_normal(
            [self.nn_output_size, self.output_size], stddev=0.1), name='net_output_weights')
        self.interface_weights = tensorflow.Variable(tensorflow.truncated_normal(
            [self.nn_output_size, self.interface_size], stddev=0.1), name='interface_weights')
        self.read_vecs_out_weight = tensorflow.Variable(tensorflow.truncated_normal(
            [self.num_heads*self.word_size, self.output_size], stddev=0.1), name='read_vector_weights')

    def allocation_weighting(self):
        """
        Retrieves the writing allocation weighting based on the usage free list
        The ‘usage’ of each location is represented as a number between 0 and 1,
        and a weighting that picks out unused locations is delivered to the write head.
        independent of the size and contents of the memory, meaning that
        DNCs can be trained to solve a task using one size of memory and later
        upgraded to a larger memory without retraining
        """
        sorted_usage_vec, free_list = tensorflow.nn.top_k(-1 * self.usage_vec, k=self.num_words)
        sorted_usage_vec *= -1
        cumprod = tensorflow.cumprod(sorted_usage_vec, axis=0, exclusive=True)
        unorder = (1-sorted_usage_vec)*cumprod
        alloc_weights = tensorflow.zeros([self.num_words])
        I = tensorflow.constant(numpy.identity(self.num_words, dtype=numpy.float32))
        for pos, idx in enumerate(tensorflow.unstack(free_list[0])):
            m = tensorflow.squeeze(tensorflow.slice(I, [idx, 0], [1, -1]))
            alloc_weights += m*unorder[0, pos]
        return tensorflow.reshape(alloc_weights, [self.num_words, 1])

    def content_lookup(self, key, str):
        """
        A key vector emitted by the controller is compared to the
        content of each location in memory according to a similarity measure
        The similarity scores determine a weighting that can be used by the read heads
        for associative recall1 or by the write head to modify an existing vector in memory.
        """
        norm_mem = tensorflow.nn.l2_normalize(self.mem_mat, 1) # N * W
        norm_key = tensorflow.nn.l2_normalize(key, 0)
        sim = tensorflow.matmul(norm_mem, norm_key, transpose_b=True)
        return tensorflow.nn.softmax(sim * str, 0)

    def run(self):
        """
        Output list of numbers (one hot encoded) by running the step function.
        """
        big_out = list()
        for t, seq in enumerate(tensorflow.unstack(self.i_data, axis=0)):
            seq = tensorflow.expand_dims(seq, 0)
            y = self.step_m(seq)
            big_out.append(y)
        return tensorflow.stack(big_out, axis=0)

    def step_m(self, x):
        """
        At every time step the controller receives input vector from dataset and emits output vector.
        It also recieves a set of read vectors from the memory matrix at the previous time step via
        the read heads. Then it emits an interface vector that defines its interactions with the memory
        at the current time step.
        """
        input = tensorflow.concat([x, tensorflow.reshape(
            self.read_vecs, [1, self.num_heads * self.word_size])], 1)
        l1_out = tensorflow.matmul(input, self.W1) + self.b1
        l1_act = tensorflow.nn.tanh(l1_out)
        l2_out = tensorflow.matmul(l1_act, self.W2) + self.b2
        l2_act = tensorflow.nn.tanh(l2_out)
        self.nn_out = tensorflow.matmul(l2_act, self.nn_out_weights)
        self.interface_vec = tensorflow.matmul(l2_act, self.interface_weights)
        partition = (tensorflow.constant([
            [0] * (self.num_heads  * self.word_size) +
            [1] * (self.num_heads) +
            [2] * (self.word_size) +
            [3] +
            [4] * (self.word_size) +
            [5] * (self.word_size) +
            [6] * (self.num_heads) +
            [7] +
            [8] +
            [9] * (self.num_heads * 3)], dtype=tensorflow.int32))
        (read_keys, read_str,   write_key,  write_str,  erase_vec,
         write_vec, free_gates, alloc_gate, write_gate, read_modes) = \
         tensorflow.dynamic_partition(self.interface_vec, partition, 10)
        read_keys = tensorflow.reshape(read_keys,[self.num_heads, self.word_size])
        read_str = 1 + tensorflow.nn.softplus(tensorflow.expand_dims(read_str, 0))
        write_key = tensorflow.expand_dims(write_key, 0)
        write_str = 1 + tensorflow.nn.softplus(tensorflow.expand_dims(write_str, 0))
        erase_vec = tensorflow.nn.sigmoid(tensorflow.expand_dims(erase_vec, 0))
        write_vec = tensorflow.expand_dims(write_vec, 0)
        free_gates = tensorflow.nn.sigmoid(tensorflow.expand_dims(free_gates, 0))
        alloc_gate = tensorflow.nn.sigmoid(alloc_gate)
        write_gate = tensorflow.nn.sigmoid(write_gate)
        read_modes = tensorflow.nn.softmax(tensorflow.reshape(read_modes, [3, self.num_heads]))
        retention_vec = tensorflow.reduce_prod(1 - free_gates * self.read_weights, reduction_indices=1)
        self.usage_vec = (
            self.usage_vec + self.write_weights - self.usage_vec * self.write_weights) * retention_vec
        alloc_weights = self.allocation_weighting()
        write_lookup_weights = self.content_lookup(write_key, write_str)
        self.write_weights = write_gate * (
            alloc_gate*alloc_weights + (1 - alloc_gate) * write_lookup_weights)
        self.mem_mat = (
            self.mem_mat * (1 - tensorflow.matmul(self.write_weights, erase_vec)) +
            (tensorflow.matmul(self.write_weights, write_vec)))
        nnweight_vec = tensorflow.matmul(self.write_weights, tensorflow.ones([1, self.num_words]))
        self.link_mat = ((
            1 - nnweight_vec - tensorflow.transpose(nnweight_vec)) * self.link_mat +
            tensorflow.matmul(self.write_weights, self.precedence_weight, transpose_b=True))
        self.link_mat *= (
            tensorflow.ones([self.num_words, self.num_words]) -
            tensorflow.constant(numpy.identity(self.num_words, dtype=numpy.float32)))
        self.precedence_weight = ((1 - tensorflow.reduce_sum(
            self.write_weights, reduction_indices=0)) * self.precedence_weight + self.write_weights)
        # 3 modes - forward, backward, content lookup
        forw_w = read_modes[2] * tensorflow.matmul(self.link_mat, self.read_weights)
        look_w = read_modes[1] * self.content_lookup(read_keys, read_str)
        back_w = read_modes[0] * tensorflow.matmul(self.link_mat, self.read_weights, transpose_a=True)
        self.read_weights = back_w + look_w + forw_w
        self.read_vecs = tensorflow.transpose(tensorflow.matmul(
            self.mem_mat, self.read_weights, transpose_a=True))
        read_vec_mut = tensorflow.matmul(tensorflow.reshape(
            self.read_vecs, [1, self.num_heads * self.word_size]), self.read_vecs_out_weight)
        return self.nn_out+read_vec_mut

def main(argv=None):
    """
    Generate the input output sequences, randomly intialized.
    """
    num_seq = 10
    seq_len = 6
    seq_width = 4
    iterations = 1000
    con = numpy.random.randint(0, seq_width, size=seq_len)
    seq = numpy.zeros((seq_len, seq_width))
    seq[numpy.arange(seq_len), con] = 1
    end = numpy.asarray([[-1] * seq_width])
    zer = numpy.zeros((seq_len, seq_width))
    graph = tensorflow.Graph()
    with graph.as_default():
        with tensorflow.Session() as sess:
            differentiable_neural_computer = DifferentiableNeuralComputer(
                input_size=seq_width, output_size=seq_width, seq_len=seq_len,
                num_words=10, word_size=4, num_heads=1)
            output = tensorflow.squeeze(differentiable_neural_computer.run())
            loss = tensorflow.reduce_mean(tensorflow.nn.sigmoid_cross_entropy_with_logits(
                logits=output, labels=dnc.o_data))
            regularizers = (
                tensorflow.nn.l2_loss(dnc.W1) + tensorflow.nn.l2_loss(dnc.W2) +
                tensorflow.nn.l2_loss(dnc.b1) + tensorflow.nn.l2_loss(dnc.b2))
            loss += 5e-4 * regularizers
            optimizer = tensorflow.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
            tensorflow.global_variables_initializer().run()
            final_i_data = numpy.concatenate((seq, zer), axis=0)
            final_o_data = numpy.concatenate((zer, seq), axis=0)
            for i in range(iterations + 1):
                feed_dict = {dnc.i_data: final_i_data, dnc.o_data: final_o_data}
                l, _, predictions = sess.run([loss, optimizer, output], feed_dict=feed_dict)
                if i % 100 == 0:
                    print(i, l)
            print(final_i_data)
            print(final_o_data)
            print(predictions)

if __name__ == '__main__':
    tensorflow.app.run()
