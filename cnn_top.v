
module cnn_top (
    input signed [7:0] image [0:783],
    input signed [7:0] conv1_weights [0:71],
    input signed [31:0] conv1_biases [0:7],
    input signed [7:0] conv2_weights [0:1151],
    input signed [31:0] conv2_biases [0:15],
    input signed [7:0] fc_weights [0:7839],
    input signed [31:0] fc_biases [0:9],
    output wire signed [63:0] scores [0:9]
);

    // Conv1: 28x28x1 -> 28x28x8
    wire signed [31:0] conv1_out [0:6271];

    // MaxPool1: 28x28x8 -> 14x14x8
    wire signed [31:0] pool1_out [0:1567];

    // Requantization 1: INT32 -> INT8
    wire signed [7:0] conv2_in [0:1567];

    // Conv2: 14x14x8 -> 14x14x16
    wire signed [31:0] conv2_out [0:3135];

    // MaxPool2: 14x14x16 -> 7x7x16
    wire signed [31:0] pool2_out [0:783];

    // Requantization 2: INT32 -> INT8
    wire signed [7:0] fc_input8 [0:783];

    // FC module expects signed INT32 inputs
    wire signed [31:0] fc_input32 [0:783];

    integer k;

    conv1_8filters_bias conv1 (
        .image(image),
        .weights(conv1_weights),
        .biases(conv1_biases),
        .output_data(conv1_out)
    );

    maxpool_2x2 pool1 (
        .input_data(conv1_out),
        .output_data(pool1_out)
    );

    requantize_array #(
        .N(1568),
        .MULTIPLIER(64'd3063111),
        .SHIFT(30)
    ) rq1 (
        .input_data(pool1_out),
        .output_data(conv2_in)
    );

    conv2_16filters conv2 (
        .input_data(conv2_in),
        .weights(conv2_weights),
        .biases(conv2_biases),
        .output_data(conv2_out)
    );

    maxpool_2x2_conv2 pool2 (
        .input_data(conv2_out),
        .output_data(pool2_out)
    );

    requantize_array #(
        .N(784),
        .MULTIPLIER(64'd2850408),
        .SHIFT(30)
    ) rq2 (
        .input_data(pool2_out),
        .output_data(fc_input8)
    );

    // Sign-extend the INT8 values for the existing FC interface.
    genvar g;
    generate
        for (g = 0; g < 784; g = g + 1) begin : extend_fc_input
            assign fc_input32[g] = {
                {24{fc_input8[g][7]}},
                fc_input8[g]
            };
        end
    endgenerate

    fc_layer fc (
        .input_data(fc_input32),
        .weights(fc_weights),
        .biases(fc_biases),
        .output_data(scores)
    );

endmodule
