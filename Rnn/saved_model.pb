��0
��
B
AddV2
x"T
y"T
z"T"
Ttype:
2	��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
=
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
�
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type
output_handle"
element_dtypetype"

shape_typetype:
2	
�
TensorListReserve
element_shape"
shape_type
num_elements

handle"
element_dtypetype"

shape_typetype:
2	
�
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint���������
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
�
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
�"serve*2.4.12v2.4.0-49-g85c8b2a817f8��-
z
dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_22/kernel
s
#dense_22/kernel/Read/ReadVariableOpReadVariableOpdense_22/kernel*
_output_shapes

: *
dtype0
r
dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_22/bias
k
!dense_22/bias/Read/ReadVariableOpReadVariableOpdense_22/bias*
_output_shapes
:*
dtype0
z
dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_23/kernel
s
#dense_23/kernel/Read/ReadVariableOpReadVariableOpdense_23/kernel*
_output_shapes

:*
dtype0
r
dense_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_23/bias
k
!dense_23/bias/Read/ReadVariableOpReadVariableOpdense_23/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
�
%simple_rnn_3/simple_rnn_cell_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *6
shared_name'%simple_rnn_3/simple_rnn_cell_3/kernel
�
9simple_rnn_3/simple_rnn_cell_3/kernel/Read/ReadVariableOpReadVariableOp%simple_rnn_3/simple_rnn_cell_3/kernel*
_output_shapes

: *
dtype0
�
/simple_rnn_3/simple_rnn_cell_3/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *@
shared_name1/simple_rnn_3/simple_rnn_cell_3/recurrent_kernel
�
Csimple_rnn_3/simple_rnn_cell_3/recurrent_kernel/Read/ReadVariableOpReadVariableOp/simple_rnn_3/simple_rnn_cell_3/recurrent_kernel*
_output_shapes

:  *
dtype0
�
#simple_rnn_3/simple_rnn_cell_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#simple_rnn_3/simple_rnn_cell_3/bias
�
7simple_rnn_3/simple_rnn_cell_3/bias/Read/ReadVariableOpReadVariableOp#simple_rnn_3/simple_rnn_cell_3/bias*
_output_shapes
: *
dtype0
�
%simple_rnn_4/simple_rnn_cell_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *6
shared_name'%simple_rnn_4/simple_rnn_cell_4/kernel
�
9simple_rnn_4/simple_rnn_cell_4/kernel/Read/ReadVariableOpReadVariableOp%simple_rnn_4/simple_rnn_cell_4/kernel*
_output_shapes

:  *
dtype0
�
/simple_rnn_4/simple_rnn_cell_4/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *@
shared_name1/simple_rnn_4/simple_rnn_cell_4/recurrent_kernel
�
Csimple_rnn_4/simple_rnn_cell_4/recurrent_kernel/Read/ReadVariableOpReadVariableOp/simple_rnn_4/simple_rnn_cell_4/recurrent_kernel*
_output_shapes

:  *
dtype0
�
#simple_rnn_4/simple_rnn_cell_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#simple_rnn_4/simple_rnn_cell_4/bias
�
7simple_rnn_4/simple_rnn_cell_4/bias/Read/ReadVariableOpReadVariableOp#simple_rnn_4/simple_rnn_cell_4/bias*
_output_shapes
: *
dtype0
�
%simple_rnn_5/simple_rnn_cell_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *6
shared_name'%simple_rnn_5/simple_rnn_cell_5/kernel
�
9simple_rnn_5/simple_rnn_cell_5/kernel/Read/ReadVariableOpReadVariableOp%simple_rnn_5/simple_rnn_cell_5/kernel*
_output_shapes

:  *
dtype0
�
/simple_rnn_5/simple_rnn_cell_5/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *@
shared_name1/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel
�
Csimple_rnn_5/simple_rnn_cell_5/recurrent_kernel/Read/ReadVariableOpReadVariableOp/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel*
_output_shapes

:  *
dtype0
�
#simple_rnn_5/simple_rnn_cell_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#simple_rnn_5/simple_rnn_cell_5/bias
�
7simple_rnn_5/simple_rnn_cell_5/bias/Read/ReadVariableOpReadVariableOp#simple_rnn_5/simple_rnn_cell_5/bias*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
�
Adam/dense_22/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_22/kernel/m
�
*Adam/dense_22/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_22/kernel/m*
_output_shapes

: *
dtype0
�
Adam/dense_22/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_22/bias/m
y
(Adam/dense_22/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_22/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_23/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_23/kernel/m
�
*Adam/dense_23/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_23/kernel/m*
_output_shapes

:*
dtype0
�
Adam/dense_23/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_23/bias/m
y
(Adam/dense_23/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_23/bias/m*
_output_shapes
:*
dtype0
�
,Adam/simple_rnn_3/simple_rnn_cell_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *=
shared_name.,Adam/simple_rnn_3/simple_rnn_cell_3/kernel/m
�
@Adam/simple_rnn_3/simple_rnn_cell_3/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_3/simple_rnn_cell_3/kernel/m*
_output_shapes

: *
dtype0
�
6Adam/simple_rnn_3/simple_rnn_cell_3/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *G
shared_name86Adam/simple_rnn_3/simple_rnn_cell_3/recurrent_kernel/m
�
JAdam/simple_rnn_3/simple_rnn_cell_3/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_3/simple_rnn_cell_3/recurrent_kernel/m*
_output_shapes

:  *
dtype0
�
*Adam/simple_rnn_3/simple_rnn_cell_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*Adam/simple_rnn_3/simple_rnn_cell_3/bias/m
�
>Adam/simple_rnn_3/simple_rnn_cell_3/bias/m/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_3/simple_rnn_cell_3/bias/m*
_output_shapes
: *
dtype0
�
,Adam/simple_rnn_4/simple_rnn_cell_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *=
shared_name.,Adam/simple_rnn_4/simple_rnn_cell_4/kernel/m
�
@Adam/simple_rnn_4/simple_rnn_cell_4/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_4/simple_rnn_cell_4/kernel/m*
_output_shapes

:  *
dtype0
�
6Adam/simple_rnn_4/simple_rnn_cell_4/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *G
shared_name86Adam/simple_rnn_4/simple_rnn_cell_4/recurrent_kernel/m
�
JAdam/simple_rnn_4/simple_rnn_cell_4/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_4/simple_rnn_cell_4/recurrent_kernel/m*
_output_shapes

:  *
dtype0
�
*Adam/simple_rnn_4/simple_rnn_cell_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*Adam/simple_rnn_4/simple_rnn_cell_4/bias/m
�
>Adam/simple_rnn_4/simple_rnn_cell_4/bias/m/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_4/simple_rnn_cell_4/bias/m*
_output_shapes
: *
dtype0
�
,Adam/simple_rnn_5/simple_rnn_cell_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *=
shared_name.,Adam/simple_rnn_5/simple_rnn_cell_5/kernel/m
�
@Adam/simple_rnn_5/simple_rnn_cell_5/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_5/simple_rnn_cell_5/kernel/m*
_output_shapes

:  *
dtype0
�
6Adam/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *G
shared_name86Adam/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel/m
�
JAdam/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel/m*
_output_shapes

:  *
dtype0
�
*Adam/simple_rnn_5/simple_rnn_cell_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*Adam/simple_rnn_5/simple_rnn_cell_5/bias/m
�
>Adam/simple_rnn_5/simple_rnn_cell_5/bias/m/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_5/simple_rnn_cell_5/bias/m*
_output_shapes
: *
dtype0
�
Adam/dense_22/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_22/kernel/v
�
*Adam/dense_22/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_22/kernel/v*
_output_shapes

: *
dtype0
�
Adam/dense_22/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_22/bias/v
y
(Adam/dense_22/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_22/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_23/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_23/kernel/v
�
*Adam/dense_23/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_23/kernel/v*
_output_shapes

:*
dtype0
�
Adam/dense_23/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_23/bias/v
y
(Adam/dense_23/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_23/bias/v*
_output_shapes
:*
dtype0
�
,Adam/simple_rnn_3/simple_rnn_cell_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *=
shared_name.,Adam/simple_rnn_3/simple_rnn_cell_3/kernel/v
�
@Adam/simple_rnn_3/simple_rnn_cell_3/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_3/simple_rnn_cell_3/kernel/v*
_output_shapes

: *
dtype0
�
6Adam/simple_rnn_3/simple_rnn_cell_3/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *G
shared_name86Adam/simple_rnn_3/simple_rnn_cell_3/recurrent_kernel/v
�
JAdam/simple_rnn_3/simple_rnn_cell_3/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_3/simple_rnn_cell_3/recurrent_kernel/v*
_output_shapes

:  *
dtype0
�
*Adam/simple_rnn_3/simple_rnn_cell_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*Adam/simple_rnn_3/simple_rnn_cell_3/bias/v
�
>Adam/simple_rnn_3/simple_rnn_cell_3/bias/v/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_3/simple_rnn_cell_3/bias/v*
_output_shapes
: *
dtype0
�
,Adam/simple_rnn_4/simple_rnn_cell_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *=
shared_name.,Adam/simple_rnn_4/simple_rnn_cell_4/kernel/v
�
@Adam/simple_rnn_4/simple_rnn_cell_4/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_4/simple_rnn_cell_4/kernel/v*
_output_shapes

:  *
dtype0
�
6Adam/simple_rnn_4/simple_rnn_cell_4/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *G
shared_name86Adam/simple_rnn_4/simple_rnn_cell_4/recurrent_kernel/v
�
JAdam/simple_rnn_4/simple_rnn_cell_4/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_4/simple_rnn_cell_4/recurrent_kernel/v*
_output_shapes

:  *
dtype0
�
*Adam/simple_rnn_4/simple_rnn_cell_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*Adam/simple_rnn_4/simple_rnn_cell_4/bias/v
�
>Adam/simple_rnn_4/simple_rnn_cell_4/bias/v/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_4/simple_rnn_cell_4/bias/v*
_output_shapes
: *
dtype0
�
,Adam/simple_rnn_5/simple_rnn_cell_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *=
shared_name.,Adam/simple_rnn_5/simple_rnn_cell_5/kernel/v
�
@Adam/simple_rnn_5/simple_rnn_cell_5/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_5/simple_rnn_cell_5/kernel/v*
_output_shapes

:  *
dtype0
�
6Adam/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *G
shared_name86Adam/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel/v
�
JAdam/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel/v*
_output_shapes

:  *
dtype0
�
*Adam/simple_rnn_5/simple_rnn_cell_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*Adam/simple_rnn_5/simple_rnn_cell_5/bias/v
�
>Adam/simple_rnn_5/simple_rnn_cell_5/bias/v/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_5/simple_rnn_cell_5/bias/v*
_output_shapes
: *
dtype0

NoOpNoOp
�K
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�K
value�KB�K B�K
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
	optimizer
regularization_losses
		variables

trainable_variables
	keras_api

signatures
R
regularization_losses
	variables
trainable_variables
	keras_api
l
cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
l
cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
l
cell

state_spec
regularization_losses
 	variables
!trainable_variables
"	keras_api
h

#kernel
$bias
%regularization_losses
&	variables
'trainable_variables
(	keras_api
h

)kernel
*bias
+regularization_losses
,	variables
-trainable_variables
.	keras_api
�
/iter

0beta_1

1beta_2
	2decay
3learning_rate#m�$m�)m�*m�4m�5m�6m�7m�8m�9m�:m�;m�<m�#v�$v�)v�*v�4v�5v�6v�7v�8v�9v�:v�;v�<v�
 
^
40
51
62
73
84
95
:6
;7
<8
#9
$10
)11
*12
^
40
51
62
73
84
95
:6
;7
<8
#9
$10
)11
*12
�
regularization_losses
=layer_metrics
>layer_regularization_losses

?layers
@metrics
Anon_trainable_variables
		variables

trainable_variables
 
 
 
 
�
regularization_losses
Blayer_metrics
Clayer_regularization_losses
Dmetrics

Elayers
Fnon_trainable_variables
	variables
trainable_variables
~

4kernel
5recurrent_kernel
6bias
Gregularization_losses
H	variables
Itrainable_variables
J	keras_api
 
 

40
51
62

40
51
62
�

Kstates
regularization_losses
Llayer_metrics
Mlayer_regularization_losses
Nmetrics

Olayers
Pnon_trainable_variables
	variables
trainable_variables
~

7kernel
8recurrent_kernel
9bias
Qregularization_losses
R	variables
Strainable_variables
T	keras_api
 
 

70
81
92

70
81
92
�

Ustates
regularization_losses
Vlayer_metrics
Wlayer_regularization_losses
Xmetrics

Ylayers
Znon_trainable_variables
	variables
trainable_variables
~

:kernel
;recurrent_kernel
<bias
[regularization_losses
\	variables
]trainable_variables
^	keras_api
 
 

:0
;1
<2

:0
;1
<2
�

_states
regularization_losses
`layer_metrics
alayer_regularization_losses
bmetrics

clayers
dnon_trainable_variables
 	variables
!trainable_variables
[Y
VARIABLE_VALUEdense_22/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_22/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

#0
$1

#0
$1
�
%regularization_losses
elayer_metrics
flayer_regularization_losses
gmetrics

hlayers
inon_trainable_variables
&	variables
'trainable_variables
[Y
VARIABLE_VALUEdense_23/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_23/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

)0
*1

)0
*1
�
+regularization_losses
jlayer_metrics
klayer_regularization_losses
lmetrics

mlayers
nnon_trainable_variables
,	variables
-trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%simple_rnn_3/simple_rnn_cell_3/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE/simple_rnn_3/simple_rnn_cell_3/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#simple_rnn_3/simple_rnn_cell_3/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%simple_rnn_4/simple_rnn_cell_4/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE/simple_rnn_4/simple_rnn_cell_4/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#simple_rnn_4/simple_rnn_cell_4/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%simple_rnn_5/simple_rnn_cell_5/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel&variables/7/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#simple_rnn_5/simple_rnn_cell_5/bias&variables/8/.ATTRIBUTES/VARIABLE_VALUE
 
 
*
0
1
2
3
4
5

o0
 
 
 
 
 
 
 

40
51
62

40
51
62
�
Gregularization_losses
player_metrics
qlayer_regularization_losses
rmetrics

slayers
tnon_trainable_variables
H	variables
Itrainable_variables
 
 
 
 

0
 
 

70
81
92

70
81
92
�
Qregularization_losses
ulayer_metrics
vlayer_regularization_losses
wmetrics

xlayers
ynon_trainable_variables
R	variables
Strainable_variables
 
 
 
 

0
 
 

:0
;1
<2

:0
;1
<2
�
[regularization_losses
zlayer_metrics
{layer_regularization_losses
|metrics

}layers
~non_trainable_variables
\	variables
]trainable_variables
 
 
 
 

0
 
 
 
 
 
 
 
 
 
 
 
7
	total

�count
�	variables
�	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

0
�1

�	variables
~|
VARIABLE_VALUEAdam/dense_22/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_22/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_23/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_23/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE,Adam/simple_rnn_3/simple_rnn_cell_3/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE6Adam/simple_rnn_3/simple_rnn_cell_3/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE*Adam/simple_rnn_3/simple_rnn_cell_3/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE,Adam/simple_rnn_4/simple_rnn_cell_4/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE6Adam/simple_rnn_4/simple_rnn_cell_4/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE*Adam/simple_rnn_4/simple_rnn_cell_4/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE,Adam/simple_rnn_5/simple_rnn_cell_5/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE6Adam/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE*Adam/simple_rnn_5/simple_rnn_cell_5/bias/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_22/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_22/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_23/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_23/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE,Adam/simple_rnn_3/simple_rnn_cell_3/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE6Adam/simple_rnn_3/simple_rnn_cell_3/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE*Adam/simple_rnn_3/simple_rnn_cell_3/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE,Adam/simple_rnn_4/simple_rnn_cell_4/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE6Adam/simple_rnn_4/simple_rnn_cell_4/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE*Adam/simple_rnn_4/simple_rnn_cell_4/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE,Adam/simple_rnn_5/simple_rnn_cell_5/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE6Adam/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE*Adam/simple_rnn_5/simple_rnn_cell_5/bias/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_lambda_1_inputPlaceholder*0
_output_shapes
:������������������*
dtype0*%
shape:������������������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_lambda_1_input%simple_rnn_3/simple_rnn_cell_3/kernel#simple_rnn_3/simple_rnn_cell_3/bias/simple_rnn_3/simple_rnn_cell_3/recurrent_kernel%simple_rnn_4/simple_rnn_cell_4/kernel#simple_rnn_4/simple_rnn_cell_4/bias/simple_rnn_4/simple_rnn_cell_4/recurrent_kernel%simple_rnn_5/simple_rnn_cell_5/kernel#simple_rnn_5/simple_rnn_cell_5/bias/simple_rnn_5/simple_rnn_cell_5/recurrent_kerneldense_22/kerneldense_22/biasdense_23/kerneldense_23/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *-
f(R&
$__inference_signature_wrapper_100141
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_22/kernel/Read/ReadVariableOp!dense_22/bias/Read/ReadVariableOp#dense_23/kernel/Read/ReadVariableOp!dense_23/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp9simple_rnn_3/simple_rnn_cell_3/kernel/Read/ReadVariableOpCsimple_rnn_3/simple_rnn_cell_3/recurrent_kernel/Read/ReadVariableOp7simple_rnn_3/simple_rnn_cell_3/bias/Read/ReadVariableOp9simple_rnn_4/simple_rnn_cell_4/kernel/Read/ReadVariableOpCsimple_rnn_4/simple_rnn_cell_4/recurrent_kernel/Read/ReadVariableOp7simple_rnn_4/simple_rnn_cell_4/bias/Read/ReadVariableOp9simple_rnn_5/simple_rnn_cell_5/kernel/Read/ReadVariableOpCsimple_rnn_5/simple_rnn_cell_5/recurrent_kernel/Read/ReadVariableOp7simple_rnn_5/simple_rnn_cell_5/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_22/kernel/m/Read/ReadVariableOp(Adam/dense_22/bias/m/Read/ReadVariableOp*Adam/dense_23/kernel/m/Read/ReadVariableOp(Adam/dense_23/bias/m/Read/ReadVariableOp@Adam/simple_rnn_3/simple_rnn_cell_3/kernel/m/Read/ReadVariableOpJAdam/simple_rnn_3/simple_rnn_cell_3/recurrent_kernel/m/Read/ReadVariableOp>Adam/simple_rnn_3/simple_rnn_cell_3/bias/m/Read/ReadVariableOp@Adam/simple_rnn_4/simple_rnn_cell_4/kernel/m/Read/ReadVariableOpJAdam/simple_rnn_4/simple_rnn_cell_4/recurrent_kernel/m/Read/ReadVariableOp>Adam/simple_rnn_4/simple_rnn_cell_4/bias/m/Read/ReadVariableOp@Adam/simple_rnn_5/simple_rnn_cell_5/kernel/m/Read/ReadVariableOpJAdam/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel/m/Read/ReadVariableOp>Adam/simple_rnn_5/simple_rnn_cell_5/bias/m/Read/ReadVariableOp*Adam/dense_22/kernel/v/Read/ReadVariableOp(Adam/dense_22/bias/v/Read/ReadVariableOp*Adam/dense_23/kernel/v/Read/ReadVariableOp(Adam/dense_23/bias/v/Read/ReadVariableOp@Adam/simple_rnn_3/simple_rnn_cell_3/kernel/v/Read/ReadVariableOpJAdam/simple_rnn_3/simple_rnn_cell_3/recurrent_kernel/v/Read/ReadVariableOp>Adam/simple_rnn_3/simple_rnn_cell_3/bias/v/Read/ReadVariableOp@Adam/simple_rnn_4/simple_rnn_cell_4/kernel/v/Read/ReadVariableOpJAdam/simple_rnn_4/simple_rnn_cell_4/recurrent_kernel/v/Read/ReadVariableOp>Adam/simple_rnn_4/simple_rnn_cell_4/bias/v/Read/ReadVariableOp@Adam/simple_rnn_5/simple_rnn_cell_5/kernel/v/Read/ReadVariableOpJAdam/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel/v/Read/ReadVariableOp>Adam/simple_rnn_5/simple_rnn_cell_5/bias/v/Read/ReadVariableOpConst*;
Tin4
220	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *(
f#R!
__inference__traced_save_102773
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_22/kerneldense_22/biasdense_23/kerneldense_23/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate%simple_rnn_3/simple_rnn_cell_3/kernel/simple_rnn_3/simple_rnn_cell_3/recurrent_kernel#simple_rnn_3/simple_rnn_cell_3/bias%simple_rnn_4/simple_rnn_cell_4/kernel/simple_rnn_4/simple_rnn_cell_4/recurrent_kernel#simple_rnn_4/simple_rnn_cell_4/bias%simple_rnn_5/simple_rnn_cell_5/kernel/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel#simple_rnn_5/simple_rnn_cell_5/biastotalcountAdam/dense_22/kernel/mAdam/dense_22/bias/mAdam/dense_23/kernel/mAdam/dense_23/bias/m,Adam/simple_rnn_3/simple_rnn_cell_3/kernel/m6Adam/simple_rnn_3/simple_rnn_cell_3/recurrent_kernel/m*Adam/simple_rnn_3/simple_rnn_cell_3/bias/m,Adam/simple_rnn_4/simple_rnn_cell_4/kernel/m6Adam/simple_rnn_4/simple_rnn_cell_4/recurrent_kernel/m*Adam/simple_rnn_4/simple_rnn_cell_4/bias/m,Adam/simple_rnn_5/simple_rnn_cell_5/kernel/m6Adam/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel/m*Adam/simple_rnn_5/simple_rnn_cell_5/bias/mAdam/dense_22/kernel/vAdam/dense_22/bias/vAdam/dense_23/kernel/vAdam/dense_23/bias/v,Adam/simple_rnn_3/simple_rnn_cell_3/kernel/v6Adam/simple_rnn_3/simple_rnn_cell_3/recurrent_kernel/v*Adam/simple_rnn_3/simple_rnn_cell_3/bias/v,Adam/simple_rnn_4/simple_rnn_cell_4/kernel/v6Adam/simple_rnn_4/simple_rnn_cell_4/recurrent_kernel/v*Adam/simple_rnn_4/simple_rnn_cell_4/bias/v,Adam/simple_rnn_5/simple_rnn_cell_5/kernel/v6Adam/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel/v*Adam/simple_rnn_5/simple_rnn_cell_5/bias/v*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__traced_restore_102921��+
�
�
L__inference_simple_rnn_cell_3_layer_call_and_return_conditional_losses_97600

inputs

states"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2	
BiasAdd�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul_1/ReadVariableOpy
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2

MatMul_1k
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2
addO
TanhTanhadd:z:0*
T0*'
_output_shapes
:��������� 2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2

Identity�

Identity_1IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:���������:��������� :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_namestates
�B
�
simple_rnn_4_while_body_1002976
2simple_rnn_4_while_simple_rnn_4_while_loop_counter<
8simple_rnn_4_while_simple_rnn_4_while_maximum_iterations"
simple_rnn_4_while_placeholder$
 simple_rnn_4_while_placeholder_1$
 simple_rnn_4_while_placeholder_25
1simple_rnn_4_while_simple_rnn_4_strided_slice_1_0q
msimple_rnn_4_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_4_tensorarrayunstack_tensorlistfromtensor_0I
Esimple_rnn_4_while_simple_rnn_cell_4_matmul_readvariableop_resource_0J
Fsimple_rnn_4_while_simple_rnn_cell_4_biasadd_readvariableop_resource_0K
Gsimple_rnn_4_while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0
simple_rnn_4_while_identity!
simple_rnn_4_while_identity_1!
simple_rnn_4_while_identity_2!
simple_rnn_4_while_identity_3!
simple_rnn_4_while_identity_43
/simple_rnn_4_while_simple_rnn_4_strided_slice_1o
ksimple_rnn_4_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_4_tensorarrayunstack_tensorlistfromtensorG
Csimple_rnn_4_while_simple_rnn_cell_4_matmul_readvariableop_resourceH
Dsimple_rnn_4_while_simple_rnn_cell_4_biasadd_readvariableop_resourceI
Esimple_rnn_4_while_simple_rnn_cell_4_matmul_1_readvariableop_resource��;simple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp�:simple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOp�<simple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp�
Dsimple_rnn_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2F
Dsimple_rnn_4/while/TensorArrayV2Read/TensorListGetItem/element_shape�
6simple_rnn_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_4_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_4_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_4_while_placeholderMsimple_rnn_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:��������� *
element_dtype028
6simple_rnn_4/while/TensorArrayV2Read/TensorListGetItem�
:simple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOpEsimple_rnn_4_while_simple_rnn_cell_4_matmul_readvariableop_resource_0*
_output_shapes

:  *
dtype02<
:simple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOp�
+simple_rnn_4/while/simple_rnn_cell_4/MatMulMatMul=simple_rnn_4/while/TensorArrayV2Read/TensorListGetItem:item:0Bsimple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2-
+simple_rnn_4/while/simple_rnn_cell_4/MatMul�
;simple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOpFsimple_rnn_4_while_simple_rnn_cell_4_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype02=
;simple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp�
,simple_rnn_4/while/simple_rnn_cell_4/BiasAddBiasAdd5simple_rnn_4/while/simple_rnn_cell_4/MatMul:product:0Csimple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2.
,simple_rnn_4/while/simple_rnn_cell_4/BiasAdd�
<simple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOpGsimple_rnn_4_while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype02>
<simple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp�
-simple_rnn_4/while/simple_rnn_cell_4/MatMul_1MatMul simple_rnn_4_while_placeholder_2Dsimple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2/
-simple_rnn_4/while/simple_rnn_cell_4/MatMul_1�
(simple_rnn_4/while/simple_rnn_cell_4/addAddV25simple_rnn_4/while/simple_rnn_cell_4/BiasAdd:output:07simple_rnn_4/while/simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2*
(simple_rnn_4/while/simple_rnn_cell_4/add�
)simple_rnn_4/while/simple_rnn_cell_4/TanhTanh,simple_rnn_4/while/simple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:��������� 2+
)simple_rnn_4/while/simple_rnn_cell_4/Tanh�
7simple_rnn_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_4_while_placeholder_1simple_rnn_4_while_placeholder-simple_rnn_4/while/simple_rnn_cell_4/Tanh:y:0*
_output_shapes
: *
element_dtype029
7simple_rnn_4/while/TensorArrayV2Write/TensorListSetItemv
simple_rnn_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_4/while/add/y�
simple_rnn_4/while/addAddV2simple_rnn_4_while_placeholder!simple_rnn_4/while/add/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_4/while/addz
simple_rnn_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_4/while/add_1/y�
simple_rnn_4/while/add_1AddV22simple_rnn_4_while_simple_rnn_4_while_loop_counter#simple_rnn_4/while/add_1/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_4/while/add_1�
simple_rnn_4/while/IdentityIdentitysimple_rnn_4/while/add_1:z:0<^simple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp;^simple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOp=^simple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_4/while/Identity�
simple_rnn_4/while/Identity_1Identity8simple_rnn_4_while_simple_rnn_4_while_maximum_iterations<^simple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp;^simple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOp=^simple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_4/while/Identity_1�
simple_rnn_4/while/Identity_2Identitysimple_rnn_4/while/add:z:0<^simple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp;^simple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOp=^simple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_4/while/Identity_2�
simple_rnn_4/while/Identity_3IdentityGsimple_rnn_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0<^simple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp;^simple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOp=^simple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_4/while/Identity_3�
simple_rnn_4/while/Identity_4Identity-simple_rnn_4/while/simple_rnn_cell_4/Tanh:y:0<^simple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp;^simple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOp=^simple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2
simple_rnn_4/while/Identity_4"C
simple_rnn_4_while_identity$simple_rnn_4/while/Identity:output:0"G
simple_rnn_4_while_identity_1&simple_rnn_4/while/Identity_1:output:0"G
simple_rnn_4_while_identity_2&simple_rnn_4/while/Identity_2:output:0"G
simple_rnn_4_while_identity_3&simple_rnn_4/while/Identity_3:output:0"G
simple_rnn_4_while_identity_4&simple_rnn_4/while/Identity_4:output:0"d
/simple_rnn_4_while_simple_rnn_4_strided_slice_11simple_rnn_4_while_simple_rnn_4_strided_slice_1_0"�
Dsimple_rnn_4_while_simple_rnn_cell_4_biasadd_readvariableop_resourceFsimple_rnn_4_while_simple_rnn_cell_4_biasadd_readvariableop_resource_0"�
Esimple_rnn_4_while_simple_rnn_cell_4_matmul_1_readvariableop_resourceGsimple_rnn_4_while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0"�
Csimple_rnn_4_while_simple_rnn_cell_4_matmul_readvariableop_resourceEsimple_rnn_4_while_simple_rnn_cell_4_matmul_readvariableop_resource_0"�
ksimple_rnn_4_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_4_tensorarrayunstack_tensorlistfromtensormsimple_rnn_4_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_4_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :��������� : : :::2z
;simple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp;simple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp2x
:simple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOp:simple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOp2|
<simple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp<simple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�
�
L__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_98641

inputs

states"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2	
BiasAdd�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul_1/ReadVariableOpy
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2

MatMul_1k
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2
addO
TanhTanhadd:z:0*
T0*'
_output_shapes
:��������� 2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2

Identity�

Identity_1IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:��������� :��������� :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_namestates
�H
�
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_101761
inputs_04
0simple_rnn_cell_4_matmul_readvariableop_resource5
1simple_rnn_cell_4_biasadd_readvariableop_resource6
2simple_rnn_cell_4_matmul_1_readvariableop_resource
identity��(simple_rnn_cell_4/BiasAdd/ReadVariableOp�'simple_rnn_cell_4/MatMul/ReadVariableOp�)simple_rnn_cell_4/MatMul_1/ReadVariableOp�whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
strided_slice_2�
'simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_4_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02)
'simple_rnn_cell_4/MatMul/ReadVariableOp�
simple_rnn_cell_4/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_4/MatMul�
(simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(simple_rnn_cell_4/BiasAdd/ReadVariableOp�
simple_rnn_cell_4/BiasAddBiasAdd"simple_rnn_cell_4/MatMul:product:00simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_4/BiasAdd�
)simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_4_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype02+
)simple_rnn_cell_4/MatMul_1/ReadVariableOp�
simple_rnn_cell_4/MatMul_1MatMulzeros:output:01simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_4/MatMul_1�
simple_rnn_cell_4/addAddV2"simple_rnn_cell_4/BiasAdd:output:0$simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_4/add�
simple_rnn_cell_4/TanhTanhsimple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_4/Tanh�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_4_matmul_readvariableop_resource1simple_rnn_cell_4_biasadd_readvariableop_resource2simple_rnn_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :��������� : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_101695*
condR
while_cond_101694*8
output_shapes'
%: : : : :��������� : : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
transpose_1�
IdentityIdentitytranspose_1:y:0)^simple_rnn_cell_4/BiasAdd/ReadVariableOp(^simple_rnn_cell_4/MatMul/ReadVariableOp*^simple_rnn_cell_4/MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :������������������ 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������ :::2T
(simple_rnn_cell_4/BiasAdd/ReadVariableOp(simple_rnn_cell_4/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_4/MatMul/ReadVariableOp'simple_rnn_cell_4/MatMul/ReadVariableOp2V
)simple_rnn_cell_4/MatMul_1/ReadVariableOp)simple_rnn_cell_4/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������ 
"
_user_specified_name
inputs/0
�H
�
G__inference_simple_rnn_4_layer_call_and_return_conditional_losses_99592

inputs4
0simple_rnn_cell_4_matmul_readvariableop_resource5
1simple_rnn_cell_4_biasadd_readvariableop_resource6
2simple_rnn_cell_4_matmul_1_readvariableop_resource
identity��(simple_rnn_cell_4/BiasAdd/ReadVariableOp�'simple_rnn_cell_4/MatMul/ReadVariableOp�)simple_rnn_cell_4/MatMul_1/ReadVariableOp�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
strided_slice_2�
'simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_4_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02)
'simple_rnn_cell_4/MatMul/ReadVariableOp�
simple_rnn_cell_4/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_4/MatMul�
(simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(simple_rnn_cell_4/BiasAdd/ReadVariableOp�
simple_rnn_cell_4/BiasAddBiasAdd"simple_rnn_cell_4/MatMul:product:00simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_4/BiasAdd�
)simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_4_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype02+
)simple_rnn_cell_4/MatMul_1/ReadVariableOp�
simple_rnn_cell_4/MatMul_1MatMulzeros:output:01simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_4/MatMul_1�
simple_rnn_cell_4/addAddV2"simple_rnn_cell_4/BiasAdd:output:0$simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_4/add�
simple_rnn_cell_4/TanhTanhsimple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_4/Tanh�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_4_matmul_readvariableop_resource1simple_rnn_cell_4_biasadd_readvariableop_resource2simple_rnn_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :��������� : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_99526*
condR
while_cond_99525*8
output_shapes'
%: : : : :��������� : : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
transpose_1�
IdentityIdentitytranspose_1:y:0)^simple_rnn_cell_4/BiasAdd/ReadVariableOp(^simple_rnn_cell_4/MatMul/ReadVariableOp*^simple_rnn_cell_4/MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :������������������ 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������ :::2T
(simple_rnn_cell_4/BiasAdd/ReadVariableOp(simple_rnn_cell_4/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_4/MatMul/ReadVariableOp'simple_rnn_cell_4/MatMul/ReadVariableOp2V
)simple_rnn_cell_4/MatMul_1/ReadVariableOp)simple_rnn_cell_4/MatMul_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�<
�
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_98054

inputs
simple_rnn_cell_3_97979
simple_rnn_cell_3_97981
simple_rnn_cell_3_97983
identity��)simple_rnn_cell_3/StatefulPartitionedCall�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
)simple_rnn_cell_3/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_3_97979simple_rnn_cell_3_97981simple_rnn_cell_3_97983*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:��������� :��������� *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_simple_rnn_cell_3_layer_call_and_return_conditional_losses_976172+
)simple_rnn_cell_3/StatefulPartitionedCall�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_3_97979simple_rnn_cell_3_97981simple_rnn_cell_3_97983*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :��������� : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_97991*
condR
while_cond_97990*8
output_shapes'
%: : : : :��������� : : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
transpose_1�
IdentityIdentitytranspose_1:y:0*^simple_rnn_cell_3/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :������������������ 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::2V
)simple_rnn_cell_3/StatefulPartitionedCall)simple_rnn_cell_3/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
�
-__inference_simple_rnn_4_layer_call_fn_101638

inputs
unknown
	unknown_0
	unknown_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_simple_rnn_4_layer_call_and_return_conditional_losses_994802
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :������������������ 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������ :::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�!
�
H__inference_sequential_6_layer_call_and_return_conditional_losses_100004

inputs
simple_rnn_3_99972
simple_rnn_3_99974
simple_rnn_3_99976
simple_rnn_4_99979
simple_rnn_4_99981
simple_rnn_4_99983
simple_rnn_5_99986
simple_rnn_5_99988
simple_rnn_5_99990
dense_22_99993
dense_22_99995
dense_23_99998
dense_23_100000
identity�� dense_22/StatefulPartitionedCall� dense_23/StatefulPartitionedCall�$simple_rnn_3/StatefulPartitionedCall�$simple_rnn_4/StatefulPartitionedCall�$simple_rnn_5/StatefulPartitionedCall�
lambda_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_lambda_1_layer_call_and_return_conditional_losses_990972
lambda_1/PartitionedCall�
$simple_rnn_3/StatefulPartitionedCallStatefulPartitionedCall!lambda_1/PartitionedCall:output:0simple_rnn_3_99972simple_rnn_3_99974simple_rnn_3_99976*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_992272&
$simple_rnn_3/StatefulPartitionedCall�
$simple_rnn_4/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_3/StatefulPartitionedCall:output:0simple_rnn_4_99979simple_rnn_4_99981simple_rnn_4_99983*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_simple_rnn_4_layer_call_and_return_conditional_losses_994802&
$simple_rnn_4/StatefulPartitionedCall�
$simple_rnn_5/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_4/StatefulPartitionedCall:output:0simple_rnn_5_99986simple_rnn_5_99988simple_rnn_5_99990*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_simple_rnn_5_layer_call_and_return_conditional_losses_997332&
$simple_rnn_5/StatefulPartitionedCall�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_5/StatefulPartitionedCall:output:0dense_22_99993dense_22_99995*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_22_layer_call_and_return_conditional_losses_998862"
 dense_22/StatefulPartitionedCall�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_99998dense_23_100000*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_23_layer_call_and_return_conditional_losses_999122"
 dense_23/StatefulPartitionedCall�
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall%^simple_rnn_3/StatefulPartitionedCall%^simple_rnn_4/StatefulPartitionedCall%^simple_rnn_5/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*c
_input_shapesR
P:������������������:::::::::::::2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2L
$simple_rnn_3/StatefulPartitionedCall$simple_rnn_3/StatefulPartitionedCall2L
$simple_rnn_4/StatefulPartitionedCall$simple_rnn_4/StatefulPartitionedCall2L
$simple_rnn_5/StatefulPartitionedCall$simple_rnn_5/StatefulPartitionedCall:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�B
�
simple_rnn_3_while_body_1005326
2simple_rnn_3_while_simple_rnn_3_while_loop_counter<
8simple_rnn_3_while_simple_rnn_3_while_maximum_iterations"
simple_rnn_3_while_placeholder$
 simple_rnn_3_while_placeholder_1$
 simple_rnn_3_while_placeholder_25
1simple_rnn_3_while_simple_rnn_3_strided_slice_1_0q
msimple_rnn_3_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_3_tensorarrayunstack_tensorlistfromtensor_0I
Esimple_rnn_3_while_simple_rnn_cell_3_matmul_readvariableop_resource_0J
Fsimple_rnn_3_while_simple_rnn_cell_3_biasadd_readvariableop_resource_0K
Gsimple_rnn_3_while_simple_rnn_cell_3_matmul_1_readvariableop_resource_0
simple_rnn_3_while_identity!
simple_rnn_3_while_identity_1!
simple_rnn_3_while_identity_2!
simple_rnn_3_while_identity_3!
simple_rnn_3_while_identity_43
/simple_rnn_3_while_simple_rnn_3_strided_slice_1o
ksimple_rnn_3_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_3_tensorarrayunstack_tensorlistfromtensorG
Csimple_rnn_3_while_simple_rnn_cell_3_matmul_readvariableop_resourceH
Dsimple_rnn_3_while_simple_rnn_cell_3_biasadd_readvariableop_resourceI
Esimple_rnn_3_while_simple_rnn_cell_3_matmul_1_readvariableop_resource��;simple_rnn_3/while/simple_rnn_cell_3/BiasAdd/ReadVariableOp�:simple_rnn_3/while/simple_rnn_cell_3/MatMul/ReadVariableOp�<simple_rnn_3/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp�
Dsimple_rnn_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2F
Dsimple_rnn_3/while/TensorArrayV2Read/TensorListGetItem/element_shape�
6simple_rnn_3/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_3_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_3_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_3_while_placeholderMsimple_rnn_3/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype028
6simple_rnn_3/while/TensorArrayV2Read/TensorListGetItem�
:simple_rnn_3/while/simple_rnn_cell_3/MatMul/ReadVariableOpReadVariableOpEsimple_rnn_3_while_simple_rnn_cell_3_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype02<
:simple_rnn_3/while/simple_rnn_cell_3/MatMul/ReadVariableOp�
+simple_rnn_3/while/simple_rnn_cell_3/MatMulMatMul=simple_rnn_3/while/TensorArrayV2Read/TensorListGetItem:item:0Bsimple_rnn_3/while/simple_rnn_cell_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2-
+simple_rnn_3/while/simple_rnn_cell_3/MatMul�
;simple_rnn_3/while/simple_rnn_cell_3/BiasAdd/ReadVariableOpReadVariableOpFsimple_rnn_3_while_simple_rnn_cell_3_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype02=
;simple_rnn_3/while/simple_rnn_cell_3/BiasAdd/ReadVariableOp�
,simple_rnn_3/while/simple_rnn_cell_3/BiasAddBiasAdd5simple_rnn_3/while/simple_rnn_cell_3/MatMul:product:0Csimple_rnn_3/while/simple_rnn_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2.
,simple_rnn_3/while/simple_rnn_cell_3/BiasAdd�
<simple_rnn_3/while/simple_rnn_cell_3/MatMul_1/ReadVariableOpReadVariableOpGsimple_rnn_3_while_simple_rnn_cell_3_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype02>
<simple_rnn_3/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp�
-simple_rnn_3/while/simple_rnn_cell_3/MatMul_1MatMul simple_rnn_3_while_placeholder_2Dsimple_rnn_3/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2/
-simple_rnn_3/while/simple_rnn_cell_3/MatMul_1�
(simple_rnn_3/while/simple_rnn_cell_3/addAddV25simple_rnn_3/while/simple_rnn_cell_3/BiasAdd:output:07simple_rnn_3/while/simple_rnn_cell_3/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2*
(simple_rnn_3/while/simple_rnn_cell_3/add�
)simple_rnn_3/while/simple_rnn_cell_3/TanhTanh,simple_rnn_3/while/simple_rnn_cell_3/add:z:0*
T0*'
_output_shapes
:��������� 2+
)simple_rnn_3/while/simple_rnn_cell_3/Tanh�
7simple_rnn_3/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_3_while_placeholder_1simple_rnn_3_while_placeholder-simple_rnn_3/while/simple_rnn_cell_3/Tanh:y:0*
_output_shapes
: *
element_dtype029
7simple_rnn_3/while/TensorArrayV2Write/TensorListSetItemv
simple_rnn_3/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_3/while/add/y�
simple_rnn_3/while/addAddV2simple_rnn_3_while_placeholder!simple_rnn_3/while/add/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_3/while/addz
simple_rnn_3/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_3/while/add_1/y�
simple_rnn_3/while/add_1AddV22simple_rnn_3_while_simple_rnn_3_while_loop_counter#simple_rnn_3/while/add_1/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_3/while/add_1�
simple_rnn_3/while/IdentityIdentitysimple_rnn_3/while/add_1:z:0<^simple_rnn_3/while/simple_rnn_cell_3/BiasAdd/ReadVariableOp;^simple_rnn_3/while/simple_rnn_cell_3/MatMul/ReadVariableOp=^simple_rnn_3/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_3/while/Identity�
simple_rnn_3/while/Identity_1Identity8simple_rnn_3_while_simple_rnn_3_while_maximum_iterations<^simple_rnn_3/while/simple_rnn_cell_3/BiasAdd/ReadVariableOp;^simple_rnn_3/while/simple_rnn_cell_3/MatMul/ReadVariableOp=^simple_rnn_3/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_3/while/Identity_1�
simple_rnn_3/while/Identity_2Identitysimple_rnn_3/while/add:z:0<^simple_rnn_3/while/simple_rnn_cell_3/BiasAdd/ReadVariableOp;^simple_rnn_3/while/simple_rnn_cell_3/MatMul/ReadVariableOp=^simple_rnn_3/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_3/while/Identity_2�
simple_rnn_3/while/Identity_3IdentityGsimple_rnn_3/while/TensorArrayV2Write/TensorListSetItem:output_handle:0<^simple_rnn_3/while/simple_rnn_cell_3/BiasAdd/ReadVariableOp;^simple_rnn_3/while/simple_rnn_cell_3/MatMul/ReadVariableOp=^simple_rnn_3/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_3/while/Identity_3�
simple_rnn_3/while/Identity_4Identity-simple_rnn_3/while/simple_rnn_cell_3/Tanh:y:0<^simple_rnn_3/while/simple_rnn_cell_3/BiasAdd/ReadVariableOp;^simple_rnn_3/while/simple_rnn_cell_3/MatMul/ReadVariableOp=^simple_rnn_3/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2
simple_rnn_3/while/Identity_4"C
simple_rnn_3_while_identity$simple_rnn_3/while/Identity:output:0"G
simple_rnn_3_while_identity_1&simple_rnn_3/while/Identity_1:output:0"G
simple_rnn_3_while_identity_2&simple_rnn_3/while/Identity_2:output:0"G
simple_rnn_3_while_identity_3&simple_rnn_3/while/Identity_3:output:0"G
simple_rnn_3_while_identity_4&simple_rnn_3/while/Identity_4:output:0"d
/simple_rnn_3_while_simple_rnn_3_strided_slice_11simple_rnn_3_while_simple_rnn_3_strided_slice_1_0"�
Dsimple_rnn_3_while_simple_rnn_cell_3_biasadd_readvariableop_resourceFsimple_rnn_3_while_simple_rnn_cell_3_biasadd_readvariableop_resource_0"�
Esimple_rnn_3_while_simple_rnn_cell_3_matmul_1_readvariableop_resourceGsimple_rnn_3_while_simple_rnn_cell_3_matmul_1_readvariableop_resource_0"�
Csimple_rnn_3_while_simple_rnn_cell_3_matmul_readvariableop_resourceEsimple_rnn_3_while_simple_rnn_cell_3_matmul_readvariableop_resource_0"�
ksimple_rnn_3_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_3_tensorarrayunstack_tensorlistfromtensormsimple_rnn_3_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_3_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :��������� : : :::2z
;simple_rnn_3/while/simple_rnn_cell_3/BiasAdd/ReadVariableOp;simple_rnn_3/while/simple_rnn_cell_3/BiasAdd/ReadVariableOp2x
:simple_rnn_3/while/simple_rnn_cell_3/MatMul/ReadVariableOp:simple_rnn_3/while/simple_rnn_cell_3/MatMul/ReadVariableOp2|
<simple_rnn_3/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp<simple_rnn_3/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�
�
-__inference_simple_rnn_3_layer_call_fn_101403

inputs
unknown
	unknown_0
	unknown_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_993392
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :������������������ 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
`
D__inference_lambda_1_layer_call_and_return_conditional_losses_100895

inputs
identityk
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
ExpandDims/dim�

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*4
_output_shapes"
 :������������������2

ExpandDimst
IdentityIdentityExpandDims:output:0*
T0*4
_output_shapes"
 :������������������2

Identity"
identityIdentity:output:0*/
_input_shapes
:������������������:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�3
�
while_body_99161
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_3_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_3_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_3_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_3_matmul_readvariableop_resource;
7while_simple_rnn_cell_3_biasadd_readvariableop_resource<
8while_simple_rnn_cell_3_matmul_1_readvariableop_resource��.while/simple_rnn_cell_3/BiasAdd/ReadVariableOp�-while/simple_rnn_cell_3/MatMul/ReadVariableOp�/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
-while/simple_rnn_cell_3/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_3_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype02/
-while/simple_rnn_cell_3/MatMul/ReadVariableOp�
while/simple_rnn_cell_3/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2 
while/simple_rnn_cell_3/MatMul�
.while/simple_rnn_cell_3/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_3_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype020
.while/simple_rnn_cell_3/BiasAdd/ReadVariableOp�
while/simple_rnn_cell_3/BiasAddBiasAdd(while/simple_rnn_cell_3/MatMul:product:06while/simple_rnn_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2!
while/simple_rnn_cell_3/BiasAdd�
/while/simple_rnn_cell_3/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_3_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype021
/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp�
 while/simple_rnn_cell_3/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2"
 while/simple_rnn_cell_3/MatMul_1�
while/simple_rnn_cell_3/addAddV2(while/simple_rnn_cell_3/BiasAdd:output:0*while/simple_rnn_cell_3/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2
while/simple_rnn_cell_3/add�
while/simple_rnn_cell_3/TanhTanhwhile/simple_rnn_cell_3/add:z:0*
T0*'
_output_shapes
:��������� 2
while/simple_rnn_cell_3/Tanh�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_3/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_3/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_3/MatMul/ReadVariableOp0^while/simple_rnn_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_3/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_3/MatMul/ReadVariableOp0^while/simple_rnn_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_3/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_3/MatMul/ReadVariableOp0^while/simple_rnn_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_3/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_3/MatMul/ReadVariableOp0^while/simple_rnn_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity while/simple_rnn_cell_3/Tanh:y:0/^while/simple_rnn_cell_3/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_3/MatMul/ReadVariableOp0^while/simple_rnn_cell_3/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_3_biasadd_readvariableop_resource9while_simple_rnn_cell_3_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_3_matmul_1_readvariableop_resource:while_simple_rnn_cell_3_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_3_matmul_readvariableop_resource8while_simple_rnn_cell_3_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :��������� : : :::2`
.while/simple_rnn_cell_3/BiasAdd/ReadVariableOp.while/simple_rnn_cell_3/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_3/MatMul/ReadVariableOp-while/simple_rnn_cell_3/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�3
�
while_body_101561
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_4_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_4_matmul_readvariableop_resource;
7while_simple_rnn_cell_4_biasadd_readvariableop_resource<
8while_simple_rnn_cell_4_matmul_1_readvariableop_resource��.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp�-while/simple_rnn_cell_4/MatMul/ReadVariableOp�/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:��������� *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
-while/simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_4_matmul_readvariableop_resource_0*
_output_shapes

:  *
dtype02/
-while/simple_rnn_cell_4/MatMul/ReadVariableOp�
while/simple_rnn_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2 
while/simple_rnn_cell_4/MatMul�
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype020
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp�
while/simple_rnn_cell_4/BiasAddBiasAdd(while/simple_rnn_cell_4/MatMul:product:06while/simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2!
while/simple_rnn_cell_4/BiasAdd�
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype021
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp�
 while/simple_rnn_cell_4/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2"
 while/simple_rnn_cell_4/MatMul_1�
while/simple_rnn_cell_4/addAddV2(while/simple_rnn_cell_4/BiasAdd:output:0*while/simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2
while/simple_rnn_cell_4/add�
while/simple_rnn_cell_4/TanhTanhwhile/simple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:��������� 2
while/simple_rnn_cell_4/Tanh�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_4/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity while/simple_rnn_cell_4/Tanh:y:0/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_4_biasadd_readvariableop_resource9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_4_matmul_1_readvariableop_resource:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_4_matmul_readvariableop_resource8while_simple_rnn_cell_4_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :��������� : : :::2`
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_4/MatMul/ReadVariableOp-while/simple_rnn_cell_4/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�H
�
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_101873
inputs_04
0simple_rnn_cell_4_matmul_readvariableop_resource5
1simple_rnn_cell_4_biasadd_readvariableop_resource6
2simple_rnn_cell_4_matmul_1_readvariableop_resource
identity��(simple_rnn_cell_4/BiasAdd/ReadVariableOp�'simple_rnn_cell_4/MatMul/ReadVariableOp�)simple_rnn_cell_4/MatMul_1/ReadVariableOp�whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
strided_slice_2�
'simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_4_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02)
'simple_rnn_cell_4/MatMul/ReadVariableOp�
simple_rnn_cell_4/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_4/MatMul�
(simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(simple_rnn_cell_4/BiasAdd/ReadVariableOp�
simple_rnn_cell_4/BiasAddBiasAdd"simple_rnn_cell_4/MatMul:product:00simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_4/BiasAdd�
)simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_4_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype02+
)simple_rnn_cell_4/MatMul_1/ReadVariableOp�
simple_rnn_cell_4/MatMul_1MatMulzeros:output:01simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_4/MatMul_1�
simple_rnn_cell_4/addAddV2"simple_rnn_cell_4/BiasAdd:output:0$simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_4/add�
simple_rnn_cell_4/TanhTanhsimple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_4/Tanh�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_4_matmul_readvariableop_resource1simple_rnn_cell_4_biasadd_readvariableop_resource2simple_rnn_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :��������� : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_101807*
condR
while_cond_101806*8
output_shapes'
%: : : : :��������� : : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
transpose_1�
IdentityIdentitytranspose_1:y:0)^simple_rnn_cell_4/BiasAdd/ReadVariableOp(^simple_rnn_cell_4/MatMul/ReadVariableOp*^simple_rnn_cell_4/MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :������������������ 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������ :::2T
(simple_rnn_cell_4/BiasAdd/ReadVariableOp(simple_rnn_cell_4/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_4/MatMul/ReadVariableOp'simple_rnn_cell_4/MatMul/ReadVariableOp2V
)simple_rnn_cell_4/MatMul_1/ReadVariableOp)simple_rnn_cell_4/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������ 
"
_user_specified_name
inputs/0
�
�
while_cond_98385
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_98385___redundant_placeholder03
/while_while_cond_98385___redundant_placeholder13
/while_while_cond_98385___redundant_placeholder23
/while_while_cond_98385___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�
�
while_cond_99778
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_99778___redundant_placeholder03
/while_while_cond_99778___redundant_placeholder13
/while_while_cond_99778___redundant_placeholder23
/while_while_cond_99778___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�H
�
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_102253

inputs4
0simple_rnn_cell_5_matmul_readvariableop_resource5
1simple_rnn_cell_5_biasadd_readvariableop_resource6
2simple_rnn_cell_5_matmul_1_readvariableop_resource
identity��(simple_rnn_cell_5/BiasAdd/ReadVariableOp�'simple_rnn_cell_5/MatMul/ReadVariableOp�)simple_rnn_cell_5/MatMul_1/ReadVariableOp�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
strided_slice_2�
'simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_5_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02)
'simple_rnn_cell_5/MatMul/ReadVariableOp�
simple_rnn_cell_5/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_5/MatMul�
(simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(simple_rnn_cell_5/BiasAdd/ReadVariableOp�
simple_rnn_cell_5/BiasAddBiasAdd"simple_rnn_cell_5/MatMul:product:00simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_5/BiasAdd�
)simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype02+
)simple_rnn_cell_5/MatMul_1/ReadVariableOp�
simple_rnn_cell_5/MatMul_1MatMulzeros:output:01simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_5/MatMul_1�
simple_rnn_cell_5/addAddV2"simple_rnn_cell_5/BiasAdd:output:0$simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_5/add�
simple_rnn_cell_5/TanhTanhsimple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_5/Tanh�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_5_matmul_readvariableop_resource1simple_rnn_cell_5_biasadd_readvariableop_resource2simple_rnn_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :��������� : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_102187*
condR
while_cond_102186*8
output_shapes'
%: : : : :��������� : : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
transpose_1�
IdentityIdentitystrided_slice_3:output:0)^simple_rnn_cell_5/BiasAdd/ReadVariableOp(^simple_rnn_cell_5/MatMul/ReadVariableOp*^simple_rnn_cell_5/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������ :::2T
(simple_rnn_cell_5/BiasAdd/ReadVariableOp(simple_rnn_cell_5/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_5/MatMul/ReadVariableOp'simple_rnn_cell_5/MatMul/ReadVariableOp2V
)simple_rnn_cell_5/MatMul_1/ReadVariableOp)simple_rnn_cell_5/MatMul_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�
�
L__inference_simple_rnn_cell_3_layer_call_and_return_conditional_losses_97617

inputs

states"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2	
BiasAdd�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul_1/ReadVariableOpy
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2

MatMul_1k
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2
addO
TanhTanhadd:z:0*
T0*'
_output_shapes
:��������� 2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2

Identity�

Identity_1IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:���������:��������� :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_namestates
�3
�
while_body_102187
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_5_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_5_matmul_readvariableop_resource;
7while_simple_rnn_cell_5_biasadd_readvariableop_resource<
8while_simple_rnn_cell_5_matmul_1_readvariableop_resource��.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp�-while/simple_rnn_cell_5/MatMul/ReadVariableOp�/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:��������� *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
-while/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_5_matmul_readvariableop_resource_0*
_output_shapes

:  *
dtype02/
-while/simple_rnn_cell_5/MatMul/ReadVariableOp�
while/simple_rnn_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2 
while/simple_rnn_cell_5/MatMul�
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype020
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp�
while/simple_rnn_cell_5/BiasAddBiasAdd(while/simple_rnn_cell_5/MatMul:product:06while/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2!
while/simple_rnn_cell_5/BiasAdd�
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype021
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp�
 while/simple_rnn_cell_5/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2"
 while/simple_rnn_cell_5/MatMul_1�
while/simple_rnn_cell_5/addAddV2(while/simple_rnn_cell_5/BiasAdd:output:0*while/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2
while/simple_rnn_cell_5/add�
while/simple_rnn_cell_5/TanhTanhwhile/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:��������� 2
while/simple_rnn_cell_5/Tanh�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_5/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity while/simple_rnn_cell_5/Tanh:y:0/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_5_biasadd_readvariableop_resource9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_5_matmul_1_readvariableop_resource:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_5_matmul_readvariableop_resource8while_simple_rnn_cell_5_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :��������� : : :::2`
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_5/MatMul/ReadVariableOp-while/simple_rnn_cell_5/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_99413
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_99413___redundant_placeholder03
/while_while_cond_99413___redundant_placeholder13
/while_while_cond_99413___redundant_placeholder23
/while_while_cond_99413___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�	
�
2__inference_simple_rnn_cell_3_layer_call_fn_102488

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:��������� :��������� *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_simple_rnn_cell_3_layer_call_and_return_conditional_losses_976172
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:��������� 2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:��������� 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:���������:��������� :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states/0
�3
�
while_body_101807
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_4_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_4_matmul_readvariableop_resource;
7while_simple_rnn_cell_4_biasadd_readvariableop_resource<
8while_simple_rnn_cell_4_matmul_1_readvariableop_resource��.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp�-while/simple_rnn_cell_4/MatMul/ReadVariableOp�/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:��������� *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
-while/simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_4_matmul_readvariableop_resource_0*
_output_shapes

:  *
dtype02/
-while/simple_rnn_cell_4/MatMul/ReadVariableOp�
while/simple_rnn_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2 
while/simple_rnn_cell_4/MatMul�
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype020
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp�
while/simple_rnn_cell_4/BiasAddBiasAdd(while/simple_rnn_cell_4/MatMul:product:06while/simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2!
while/simple_rnn_cell_4/BiasAdd�
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype021
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp�
 while/simple_rnn_cell_4/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2"
 while/simple_rnn_cell_4/MatMul_1�
while/simple_rnn_cell_4/addAddV2(while/simple_rnn_cell_4/BiasAdd:output:0*while/simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2
while/simple_rnn_cell_4/add�
while/simple_rnn_cell_4/TanhTanhwhile/simple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:��������� 2
while/simple_rnn_cell_4/Tanh�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_4/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity while/simple_rnn_cell_4/Tanh:y:0/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_4_biasadd_readvariableop_resource9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_4_matmul_1_readvariableop_resource:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_4_matmul_readvariableop_resource8while_simple_rnn_cell_4_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :��������� : : :::2`
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_4/MatMul/ReadVariableOp-while/simple_rnn_cell_4/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�	
�
$__inference_signature_wrapper_100141
lambda_1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllambda_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *)
f$R"
 __inference__wrapped_model_975512
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*c
_input_shapesR
P:������������������:::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
0
_output_shapes
:������������������
(
_user_specified_namelambda_1_input
�
�
M__inference_simple_rnn_cell_3_layer_call_and_return_conditional_losses_102443

inputs
states_0"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2	
BiasAdd�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2

MatMul_1k
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2
addO
TanhTanhadd:z:0*
T0*'
_output_shapes
:��������� 2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2

Identity�

Identity_1IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:���������:��������� :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states/0
�
�
-__inference_simple_rnn_5_layer_call_fn_102141
inputs_0
unknown
	unknown_0
	unknown_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_simple_rnn_5_layer_call_and_return_conditional_losses_990782
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������ :::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������ 
"
_user_specified_name
inputs/0
�
�
*sequential_6_simple_rnn_5_while_cond_97471P
Lsequential_6_simple_rnn_5_while_sequential_6_simple_rnn_5_while_loop_counterV
Rsequential_6_simple_rnn_5_while_sequential_6_simple_rnn_5_while_maximum_iterations/
+sequential_6_simple_rnn_5_while_placeholder1
-sequential_6_simple_rnn_5_while_placeholder_11
-sequential_6_simple_rnn_5_while_placeholder_2R
Nsequential_6_simple_rnn_5_while_less_sequential_6_simple_rnn_5_strided_slice_1g
csequential_6_simple_rnn_5_while_sequential_6_simple_rnn_5_while_cond_97471___redundant_placeholder0g
csequential_6_simple_rnn_5_while_sequential_6_simple_rnn_5_while_cond_97471___redundant_placeholder1g
csequential_6_simple_rnn_5_while_sequential_6_simple_rnn_5_while_cond_97471___redundant_placeholder2g
csequential_6_simple_rnn_5_while_sequential_6_simple_rnn_5_while_cond_97471___redundant_placeholder3,
(sequential_6_simple_rnn_5_while_identity
�
$sequential_6/simple_rnn_5/while/LessLess+sequential_6_simple_rnn_5_while_placeholderNsequential_6_simple_rnn_5_while_less_sequential_6_simple_rnn_5_strided_slice_1*
T0*
_output_shapes
: 2&
$sequential_6/simple_rnn_5/while/Less�
(sequential_6/simple_rnn_5/while/IdentityIdentity(sequential_6/simple_rnn_5/while/Less:z:0*
T0
*
_output_shapes
: 2*
(sequential_6/simple_rnn_5/while/Identity"]
(sequential_6_simple_rnn_5_while_identity1sequential_6/simple_rnn_5/while/Identity:output:0*@
_input_shapes/
-: : : : :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�#
�
while_body_97991
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0#
while_simple_rnn_cell_3_98013_0#
while_simple_rnn_cell_3_98015_0#
while_simple_rnn_cell_3_98017_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor!
while_simple_rnn_cell_3_98013!
while_simple_rnn_cell_3_98015!
while_simple_rnn_cell_3_98017��/while/simple_rnn_cell_3/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
/while/simple_rnn_cell_3/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_simple_rnn_cell_3_98013_0while_simple_rnn_cell_3_98015_0while_simple_rnn_cell_3_98017_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:��������� :��������� *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_simple_rnn_cell_3_layer_call_and_return_conditional_losses_9761721
/while/simple_rnn_cell_3/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder8while/simple_rnn_cell_3/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:00^while/simple_rnn_cell_3/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations0^while/simple_rnn_cell_3/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:00^while/simple_rnn_cell_3/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/simple_rnn_cell_3/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity8while/simple_rnn_cell_3/StatefulPartitionedCall:output:10^while/simple_rnn_cell_3/StatefulPartitionedCall*
T0*'
_output_shapes
:��������� 2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"@
while_simple_rnn_cell_3_98013while_simple_rnn_cell_3_98013_0"@
while_simple_rnn_cell_3_98015while_simple_rnn_cell_3_98015_0"@
while_simple_rnn_cell_3_98017while_simple_rnn_cell_3_98017_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :��������� : : :::2b
/while/simple_rnn_cell_3/StatefulPartitionedCall/while/simple_rnn_cell_3/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�
`
D__inference_lambda_1_layer_call_and_return_conditional_losses_100901

inputs
identityk
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
ExpandDims/dim�

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*4
_output_shapes"
 :������������������2

ExpandDimst
IdentityIdentityExpandDims:output:0*
T0*4
_output_shapes"
 :������������������2

Identity"
identityIdentity:output:0*/
_input_shapes
:������������������:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
�
*sequential_6_simple_rnn_3_while_cond_97255P
Lsequential_6_simple_rnn_3_while_sequential_6_simple_rnn_3_while_loop_counterV
Rsequential_6_simple_rnn_3_while_sequential_6_simple_rnn_3_while_maximum_iterations/
+sequential_6_simple_rnn_3_while_placeholder1
-sequential_6_simple_rnn_3_while_placeholder_11
-sequential_6_simple_rnn_3_while_placeholder_2R
Nsequential_6_simple_rnn_3_while_less_sequential_6_simple_rnn_3_strided_slice_1g
csequential_6_simple_rnn_3_while_sequential_6_simple_rnn_3_while_cond_97255___redundant_placeholder0g
csequential_6_simple_rnn_3_while_sequential_6_simple_rnn_3_while_cond_97255___redundant_placeholder1g
csequential_6_simple_rnn_3_while_sequential_6_simple_rnn_3_while_cond_97255___redundant_placeholder2g
csequential_6_simple_rnn_3_while_sequential_6_simple_rnn_3_while_cond_97255___redundant_placeholder3,
(sequential_6_simple_rnn_3_while_identity
�
$sequential_6/simple_rnn_3/while/LessLess+sequential_6_simple_rnn_3_while_placeholderNsequential_6_simple_rnn_3_while_less_sequential_6_simple_rnn_3_strided_slice_1*
T0*
_output_shapes
: 2&
$sequential_6/simple_rnn_3/while/Less�
(sequential_6/simple_rnn_3/while/IdentityIdentity(sequential_6/simple_rnn_3/while/Less:z:0*
T0
*
_output_shapes
: 2*
(sequential_6/simple_rnn_3/while/Identity"]
(sequential_6_simple_rnn_3_while_identity1sequential_6/simple_rnn_3/while/Identity:output:0*@
_input_shapes/
-: : : : :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�
�
-__inference_simple_rnn_5_layer_call_fn_102130
inputs_0
unknown
	unknown_0
	unknown_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_simple_rnn_5_layer_call_and_return_conditional_losses_989612
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������ :::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������ 
"
_user_specified_name
inputs/0
�
�
while_cond_98502
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_98502___redundant_placeholder03
/while_while_cond_98502___redundant_placeholder13
/while_while_cond_98502___redundant_placeholder23
/while_while_cond_98502___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�B
�
simple_rnn_5_while_body_1007486
2simple_rnn_5_while_simple_rnn_5_while_loop_counter<
8simple_rnn_5_while_simple_rnn_5_while_maximum_iterations"
simple_rnn_5_while_placeholder$
 simple_rnn_5_while_placeholder_1$
 simple_rnn_5_while_placeholder_25
1simple_rnn_5_while_simple_rnn_5_strided_slice_1_0q
msimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0I
Esimple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resource_0J
Fsimple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resource_0K
Gsimple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0
simple_rnn_5_while_identity!
simple_rnn_5_while_identity_1!
simple_rnn_5_while_identity_2!
simple_rnn_5_while_identity_3!
simple_rnn_5_while_identity_43
/simple_rnn_5_while_simple_rnn_5_strided_slice_1o
ksimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensorG
Csimple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resourceH
Dsimple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resourceI
Esimple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resource��;simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp�:simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp�<simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp�
Dsimple_rnn_5/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2F
Dsimple_rnn_5/while/TensorArrayV2Read/TensorListGetItem/element_shape�
6simple_rnn_5/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_5_while_placeholderMsimple_rnn_5/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:��������� *
element_dtype028
6simple_rnn_5/while/TensorArrayV2Read/TensorListGetItem�
:simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOpEsimple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resource_0*
_output_shapes

:  *
dtype02<
:simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp�
+simple_rnn_5/while/simple_rnn_cell_5/MatMulMatMul=simple_rnn_5/while/TensorArrayV2Read/TensorListGetItem:item:0Bsimple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2-
+simple_rnn_5/while/simple_rnn_cell_5/MatMul�
;simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOpFsimple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype02=
;simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp�
,simple_rnn_5/while/simple_rnn_cell_5/BiasAddBiasAdd5simple_rnn_5/while/simple_rnn_cell_5/MatMul:product:0Csimple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2.
,simple_rnn_5/while/simple_rnn_cell_5/BiasAdd�
<simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOpGsimple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype02>
<simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp�
-simple_rnn_5/while/simple_rnn_cell_5/MatMul_1MatMul simple_rnn_5_while_placeholder_2Dsimple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2/
-simple_rnn_5/while/simple_rnn_cell_5/MatMul_1�
(simple_rnn_5/while/simple_rnn_cell_5/addAddV25simple_rnn_5/while/simple_rnn_cell_5/BiasAdd:output:07simple_rnn_5/while/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2*
(simple_rnn_5/while/simple_rnn_cell_5/add�
)simple_rnn_5/while/simple_rnn_cell_5/TanhTanh,simple_rnn_5/while/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:��������� 2+
)simple_rnn_5/while/simple_rnn_cell_5/Tanh�
7simple_rnn_5/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_5_while_placeholder_1simple_rnn_5_while_placeholder-simple_rnn_5/while/simple_rnn_cell_5/Tanh:y:0*
_output_shapes
: *
element_dtype029
7simple_rnn_5/while/TensorArrayV2Write/TensorListSetItemv
simple_rnn_5/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_5/while/add/y�
simple_rnn_5/while/addAddV2simple_rnn_5_while_placeholder!simple_rnn_5/while/add/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_5/while/addz
simple_rnn_5/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_5/while/add_1/y�
simple_rnn_5/while/add_1AddV22simple_rnn_5_while_simple_rnn_5_while_loop_counter#simple_rnn_5/while/add_1/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_5/while/add_1�
simple_rnn_5/while/IdentityIdentitysimple_rnn_5/while/add_1:z:0<^simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp;^simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp=^simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_5/while/Identity�
simple_rnn_5/while/Identity_1Identity8simple_rnn_5_while_simple_rnn_5_while_maximum_iterations<^simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp;^simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp=^simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_5/while/Identity_1�
simple_rnn_5/while/Identity_2Identitysimple_rnn_5/while/add:z:0<^simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp;^simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp=^simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_5/while/Identity_2�
simple_rnn_5/while/Identity_3IdentityGsimple_rnn_5/while/TensorArrayV2Write/TensorListSetItem:output_handle:0<^simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp;^simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp=^simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_5/while/Identity_3�
simple_rnn_5/while/Identity_4Identity-simple_rnn_5/while/simple_rnn_cell_5/Tanh:y:0<^simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp;^simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp=^simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2
simple_rnn_5/while/Identity_4"C
simple_rnn_5_while_identity$simple_rnn_5/while/Identity:output:0"G
simple_rnn_5_while_identity_1&simple_rnn_5/while/Identity_1:output:0"G
simple_rnn_5_while_identity_2&simple_rnn_5/while/Identity_2:output:0"G
simple_rnn_5_while_identity_3&simple_rnn_5/while/Identity_3:output:0"G
simple_rnn_5_while_identity_4&simple_rnn_5/while/Identity_4:output:0"d
/simple_rnn_5_while_simple_rnn_5_strided_slice_11simple_rnn_5_while_simple_rnn_5_strided_slice_1_0"�
Dsimple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resourceFsimple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resource_0"�
Esimple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resourceGsimple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0"�
Csimple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resourceEsimple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resource_0"�
ksimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensormsimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :��������� : : :::2z
;simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp;simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp2x
:simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp:simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp2|
<simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp<simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�!
�
G__inference_sequential_6_layer_call_and_return_conditional_losses_99929
lambda_1_input
simple_rnn_3_99362
simple_rnn_3_99364
simple_rnn_3_99366
simple_rnn_4_99615
simple_rnn_4_99617
simple_rnn_4_99619
simple_rnn_5_99868
simple_rnn_5_99870
simple_rnn_5_99872
dense_22_99897
dense_22_99899
dense_23_99923
dense_23_99925
identity�� dense_22/StatefulPartitionedCall� dense_23/StatefulPartitionedCall�$simple_rnn_3/StatefulPartitionedCall�$simple_rnn_4/StatefulPartitionedCall�$simple_rnn_5/StatefulPartitionedCall�
lambda_1/PartitionedCallPartitionedCalllambda_1_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_lambda_1_layer_call_and_return_conditional_losses_990972
lambda_1/PartitionedCall�
$simple_rnn_3/StatefulPartitionedCallStatefulPartitionedCall!lambda_1/PartitionedCall:output:0simple_rnn_3_99362simple_rnn_3_99364simple_rnn_3_99366*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_992272&
$simple_rnn_3/StatefulPartitionedCall�
$simple_rnn_4/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_3/StatefulPartitionedCall:output:0simple_rnn_4_99615simple_rnn_4_99617simple_rnn_4_99619*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_simple_rnn_4_layer_call_and_return_conditional_losses_994802&
$simple_rnn_4/StatefulPartitionedCall�
$simple_rnn_5/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_4/StatefulPartitionedCall:output:0simple_rnn_5_99868simple_rnn_5_99870simple_rnn_5_99872*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_simple_rnn_5_layer_call_and_return_conditional_losses_997332&
$simple_rnn_5/StatefulPartitionedCall�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_5/StatefulPartitionedCall:output:0dense_22_99897dense_22_99899*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_22_layer_call_and_return_conditional_losses_998862"
 dense_22/StatefulPartitionedCall�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_99923dense_23_99925*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_23_layer_call_and_return_conditional_losses_999122"
 dense_23/StatefulPartitionedCall�
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall%^simple_rnn_3/StatefulPartitionedCall%^simple_rnn_4/StatefulPartitionedCall%^simple_rnn_5/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*c
_input_shapesR
P:������������������:::::::::::::2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2L
$simple_rnn_3/StatefulPartitionedCall$simple_rnn_3/StatefulPartitionedCall2L
$simple_rnn_4/StatefulPartitionedCall$simple_rnn_4/StatefulPartitionedCall2L
$simple_rnn_5/StatefulPartitionedCall$simple_rnn_5/StatefulPartitionedCall:` \
0
_output_shapes
:������������������
(
_user_specified_namelambda_1_input
�H
�
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_99227

inputs4
0simple_rnn_cell_3_matmul_readvariableop_resource5
1simple_rnn_cell_3_biasadd_readvariableop_resource6
2simple_rnn_cell_3_matmul_1_readvariableop_resource
identity��(simple_rnn_cell_3/BiasAdd/ReadVariableOp�'simple_rnn_cell_3/MatMul/ReadVariableOp�)simple_rnn_cell_3/MatMul_1/ReadVariableOp�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
'simple_rnn_cell_3/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02)
'simple_rnn_cell_3/MatMul/ReadVariableOp�
simple_rnn_cell_3/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_3/MatMul�
(simple_rnn_cell_3/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(simple_rnn_cell_3/BiasAdd/ReadVariableOp�
simple_rnn_cell_3/BiasAddBiasAdd"simple_rnn_cell_3/MatMul:product:00simple_rnn_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_3/BiasAdd�
)simple_rnn_cell_3/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_3_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype02+
)simple_rnn_cell_3/MatMul_1/ReadVariableOp�
simple_rnn_cell_3/MatMul_1MatMulzeros:output:01simple_rnn_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_3/MatMul_1�
simple_rnn_cell_3/addAddV2"simple_rnn_cell_3/BiasAdd:output:0$simple_rnn_cell_3/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_3/add�
simple_rnn_cell_3/TanhTanhsimple_rnn_cell_3/add:z:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_3/Tanh�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_3_matmul_readvariableop_resource1simple_rnn_cell_3_biasadd_readvariableop_resource2simple_rnn_cell_3_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :��������� : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_99161*
condR
while_cond_99160*8
output_shapes'
%: : : : :��������� : : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
transpose_1�
IdentityIdentitytranspose_1:y:0)^simple_rnn_cell_3/BiasAdd/ReadVariableOp(^simple_rnn_cell_3/MatMul/ReadVariableOp*^simple_rnn_cell_3/MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :������������������ 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::2T
(simple_rnn_cell_3/BiasAdd/ReadVariableOp(simple_rnn_cell_3/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_3/MatMul/ReadVariableOp'simple_rnn_cell_3/MatMul/ReadVariableOp2V
)simple_rnn_cell_3/MatMul_1/ReadVariableOp)simple_rnn_cell_3/MatMul_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
�
L__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_98624

inputs

states"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2	
BiasAdd�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul_1/ReadVariableOpy
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2

MatMul_1k
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2
addO
TanhTanhadd:z:0*
T0*'
_output_shapes
:��������� 2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2

Identity�

Identity_1IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:��������� :��������� :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_namestates
�
�
while_cond_98897
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_98897___redundant_placeholder03
/while_while_cond_98897___redundant_placeholder13
/while_while_cond_98897___redundant_placeholder23
/while_while_cond_98897___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�H
�
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_102119
inputs_04
0simple_rnn_cell_5_matmul_readvariableop_resource5
1simple_rnn_cell_5_biasadd_readvariableop_resource6
2simple_rnn_cell_5_matmul_1_readvariableop_resource
identity��(simple_rnn_cell_5/BiasAdd/ReadVariableOp�'simple_rnn_cell_5/MatMul/ReadVariableOp�)simple_rnn_cell_5/MatMul_1/ReadVariableOp�whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
strided_slice_2�
'simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_5_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02)
'simple_rnn_cell_5/MatMul/ReadVariableOp�
simple_rnn_cell_5/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_5/MatMul�
(simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(simple_rnn_cell_5/BiasAdd/ReadVariableOp�
simple_rnn_cell_5/BiasAddBiasAdd"simple_rnn_cell_5/MatMul:product:00simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_5/BiasAdd�
)simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype02+
)simple_rnn_cell_5/MatMul_1/ReadVariableOp�
simple_rnn_cell_5/MatMul_1MatMulzeros:output:01simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_5/MatMul_1�
simple_rnn_cell_5/addAddV2"simple_rnn_cell_5/BiasAdd:output:0$simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_5/add�
simple_rnn_cell_5/TanhTanhsimple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_5/Tanh�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_5_matmul_readvariableop_resource1simple_rnn_cell_5_biasadd_readvariableop_resource2simple_rnn_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :��������� : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_102053*
condR
while_cond_102052*8
output_shapes'
%: : : : :��������� : : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
transpose_1�
IdentityIdentitystrided_slice_3:output:0)^simple_rnn_cell_5/BiasAdd/ReadVariableOp(^simple_rnn_cell_5/MatMul/ReadVariableOp*^simple_rnn_cell_5/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������ :::2T
(simple_rnn_cell_5/BiasAdd/ReadVariableOp(simple_rnn_cell_5/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_5/MatMul/ReadVariableOp'simple_rnn_cell_5/MatMul/ReadVariableOp2V
)simple_rnn_cell_5/MatMul_1/ReadVariableOp)simple_rnn_cell_5/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������ 
"
_user_specified_name
inputs/0
�3
�
while_body_102053
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_5_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_5_matmul_readvariableop_resource;
7while_simple_rnn_cell_5_biasadd_readvariableop_resource<
8while_simple_rnn_cell_5_matmul_1_readvariableop_resource��.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp�-while/simple_rnn_cell_5/MatMul/ReadVariableOp�/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:��������� *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
-while/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_5_matmul_readvariableop_resource_0*
_output_shapes

:  *
dtype02/
-while/simple_rnn_cell_5/MatMul/ReadVariableOp�
while/simple_rnn_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2 
while/simple_rnn_cell_5/MatMul�
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype020
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp�
while/simple_rnn_cell_5/BiasAddBiasAdd(while/simple_rnn_cell_5/MatMul:product:06while/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2!
while/simple_rnn_cell_5/BiasAdd�
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype021
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp�
 while/simple_rnn_cell_5/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2"
 while/simple_rnn_cell_5/MatMul_1�
while/simple_rnn_cell_5/addAddV2(while/simple_rnn_cell_5/BiasAdd:output:0*while/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2
while/simple_rnn_cell_5/add�
while/simple_rnn_cell_5/TanhTanhwhile/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:��������� 2
while/simple_rnn_cell_5/Tanh�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_5/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity while/simple_rnn_cell_5/Tanh:y:0/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_5_biasadd_readvariableop_resource9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_5_matmul_1_readvariableop_resource:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_5_matmul_readvariableop_resource8while_simple_rnn_cell_5_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :��������� : : :::2`
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_5/MatMul/ReadVariableOp-while/simple_rnn_cell_5/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�	
�
2__inference_simple_rnn_cell_3_layer_call_fn_102474

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:��������� :��������� *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_simple_rnn_cell_3_layer_call_and_return_conditional_losses_976002
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:��������� 2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:��������� 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:���������:��������� :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states/0
�
~
)__inference_dense_22_layer_call_fn_102407

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_22_layer_call_and_return_conditional_losses_998862
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�B
�
simple_rnn_5_while_body_1004056
2simple_rnn_5_while_simple_rnn_5_while_loop_counter<
8simple_rnn_5_while_simple_rnn_5_while_maximum_iterations"
simple_rnn_5_while_placeholder$
 simple_rnn_5_while_placeholder_1$
 simple_rnn_5_while_placeholder_25
1simple_rnn_5_while_simple_rnn_5_strided_slice_1_0q
msimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0I
Esimple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resource_0J
Fsimple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resource_0K
Gsimple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0
simple_rnn_5_while_identity!
simple_rnn_5_while_identity_1!
simple_rnn_5_while_identity_2!
simple_rnn_5_while_identity_3!
simple_rnn_5_while_identity_43
/simple_rnn_5_while_simple_rnn_5_strided_slice_1o
ksimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensorG
Csimple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resourceH
Dsimple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resourceI
Esimple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resource��;simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp�:simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp�<simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp�
Dsimple_rnn_5/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2F
Dsimple_rnn_5/while/TensorArrayV2Read/TensorListGetItem/element_shape�
6simple_rnn_5/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_5_while_placeholderMsimple_rnn_5/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:��������� *
element_dtype028
6simple_rnn_5/while/TensorArrayV2Read/TensorListGetItem�
:simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOpEsimple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resource_0*
_output_shapes

:  *
dtype02<
:simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp�
+simple_rnn_5/while/simple_rnn_cell_5/MatMulMatMul=simple_rnn_5/while/TensorArrayV2Read/TensorListGetItem:item:0Bsimple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2-
+simple_rnn_5/while/simple_rnn_cell_5/MatMul�
;simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOpFsimple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype02=
;simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp�
,simple_rnn_5/while/simple_rnn_cell_5/BiasAddBiasAdd5simple_rnn_5/while/simple_rnn_cell_5/MatMul:product:0Csimple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2.
,simple_rnn_5/while/simple_rnn_cell_5/BiasAdd�
<simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOpGsimple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype02>
<simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp�
-simple_rnn_5/while/simple_rnn_cell_5/MatMul_1MatMul simple_rnn_5_while_placeholder_2Dsimple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2/
-simple_rnn_5/while/simple_rnn_cell_5/MatMul_1�
(simple_rnn_5/while/simple_rnn_cell_5/addAddV25simple_rnn_5/while/simple_rnn_cell_5/BiasAdd:output:07simple_rnn_5/while/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2*
(simple_rnn_5/while/simple_rnn_cell_5/add�
)simple_rnn_5/while/simple_rnn_cell_5/TanhTanh,simple_rnn_5/while/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:��������� 2+
)simple_rnn_5/while/simple_rnn_cell_5/Tanh�
7simple_rnn_5/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_5_while_placeholder_1simple_rnn_5_while_placeholder-simple_rnn_5/while/simple_rnn_cell_5/Tanh:y:0*
_output_shapes
: *
element_dtype029
7simple_rnn_5/while/TensorArrayV2Write/TensorListSetItemv
simple_rnn_5/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_5/while/add/y�
simple_rnn_5/while/addAddV2simple_rnn_5_while_placeholder!simple_rnn_5/while/add/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_5/while/addz
simple_rnn_5/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_5/while/add_1/y�
simple_rnn_5/while/add_1AddV22simple_rnn_5_while_simple_rnn_5_while_loop_counter#simple_rnn_5/while/add_1/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_5/while/add_1�
simple_rnn_5/while/IdentityIdentitysimple_rnn_5/while/add_1:z:0<^simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp;^simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp=^simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_5/while/Identity�
simple_rnn_5/while/Identity_1Identity8simple_rnn_5_while_simple_rnn_5_while_maximum_iterations<^simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp;^simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp=^simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_5/while/Identity_1�
simple_rnn_5/while/Identity_2Identitysimple_rnn_5/while/add:z:0<^simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp;^simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp=^simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_5/while/Identity_2�
simple_rnn_5/while/Identity_3IdentityGsimple_rnn_5/while/TensorArrayV2Write/TensorListSetItem:output_handle:0<^simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp;^simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp=^simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_5/while/Identity_3�
simple_rnn_5/while/Identity_4Identity-simple_rnn_5/while/simple_rnn_cell_5/Tanh:y:0<^simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp;^simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp=^simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2
simple_rnn_5/while/Identity_4"C
simple_rnn_5_while_identity$simple_rnn_5/while/Identity:output:0"G
simple_rnn_5_while_identity_1&simple_rnn_5/while/Identity_1:output:0"G
simple_rnn_5_while_identity_2&simple_rnn_5/while/Identity_2:output:0"G
simple_rnn_5_while_identity_3&simple_rnn_5/while/Identity_3:output:0"G
simple_rnn_5_while_identity_4&simple_rnn_5/while/Identity_4:output:0"d
/simple_rnn_5_while_simple_rnn_5_strided_slice_11simple_rnn_5_while_simple_rnn_5_strided_slice_1_0"�
Dsimple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resourceFsimple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resource_0"�
Esimple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resourceGsimple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0"�
Csimple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resourceEsimple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resource_0"�
ksimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensormsimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :��������� : : :::2z
;simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp;simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp2x
:simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp:simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp2|
<simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp<simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�
�
L__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_98129

inputs

states"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2	
BiasAdd�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul_1/ReadVariableOpy
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2

MatMul_1k
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2
addO
TanhTanhadd:z:0*
T0*'
_output_shapes
:��������� 2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2

Identity�

Identity_1IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:��������� :��������� :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_namestates
�
�
-__inference_simple_rnn_4_layer_call_fn_101895
inputs_0
unknown
	unknown_0
	unknown_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_simple_rnn_4_layer_call_and_return_conditional_losses_985662
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :������������������ 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������ :::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������ 
"
_user_specified_name
inputs/0
�3
�
while_body_99414
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_4_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_4_matmul_readvariableop_resource;
7while_simple_rnn_cell_4_biasadd_readvariableop_resource<
8while_simple_rnn_cell_4_matmul_1_readvariableop_resource��.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp�-while/simple_rnn_cell_4/MatMul/ReadVariableOp�/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:��������� *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
-while/simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_4_matmul_readvariableop_resource_0*
_output_shapes

:  *
dtype02/
-while/simple_rnn_cell_4/MatMul/ReadVariableOp�
while/simple_rnn_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2 
while/simple_rnn_cell_4/MatMul�
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype020
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp�
while/simple_rnn_cell_4/BiasAddBiasAdd(while/simple_rnn_cell_4/MatMul:product:06while/simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2!
while/simple_rnn_cell_4/BiasAdd�
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype021
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp�
 while/simple_rnn_cell_4/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2"
 while/simple_rnn_cell_4/MatMul_1�
while/simple_rnn_cell_4/addAddV2(while/simple_rnn_cell_4/BiasAdd:output:0*while/simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2
while/simple_rnn_cell_4/add�
while/simple_rnn_cell_4/TanhTanhwhile/simple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:��������� 2
while/simple_rnn_cell_4/Tanh�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_4/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity while/simple_rnn_cell_4/Tanh:y:0/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_4_biasadd_readvariableop_resource9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_4_matmul_1_readvariableop_resource:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_4_matmul_readvariableop_resource8while_simple_rnn_cell_4_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :��������� : : :::2`
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_4/MatMul/ReadVariableOp-while/simple_rnn_cell_4/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_99014
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_99014___redundant_placeholder03
/while_while_cond_99014___redundant_placeholder13
/while_while_cond_99014___redundant_placeholder23
/while_while_cond_99014___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�H
�
H__inference_simple_rnn_3_layer_call_and_return_conditional_losses_101381

inputs4
0simple_rnn_cell_3_matmul_readvariableop_resource5
1simple_rnn_cell_3_biasadd_readvariableop_resource6
2simple_rnn_cell_3_matmul_1_readvariableop_resource
identity��(simple_rnn_cell_3/BiasAdd/ReadVariableOp�'simple_rnn_cell_3/MatMul/ReadVariableOp�)simple_rnn_cell_3/MatMul_1/ReadVariableOp�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
'simple_rnn_cell_3/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02)
'simple_rnn_cell_3/MatMul/ReadVariableOp�
simple_rnn_cell_3/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_3/MatMul�
(simple_rnn_cell_3/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(simple_rnn_cell_3/BiasAdd/ReadVariableOp�
simple_rnn_cell_3/BiasAddBiasAdd"simple_rnn_cell_3/MatMul:product:00simple_rnn_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_3/BiasAdd�
)simple_rnn_cell_3/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_3_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype02+
)simple_rnn_cell_3/MatMul_1/ReadVariableOp�
simple_rnn_cell_3/MatMul_1MatMulzeros:output:01simple_rnn_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_3/MatMul_1�
simple_rnn_cell_3/addAddV2"simple_rnn_cell_3/BiasAdd:output:0$simple_rnn_cell_3/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_3/add�
simple_rnn_cell_3/TanhTanhsimple_rnn_cell_3/add:z:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_3/Tanh�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_3_matmul_readvariableop_resource1simple_rnn_cell_3_biasadd_readvariableop_resource2simple_rnn_cell_3_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :��������� : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_101315*
condR
while_cond_101314*8
output_shapes'
%: : : : :��������� : : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
transpose_1�
IdentityIdentitytranspose_1:y:0)^simple_rnn_cell_3/BiasAdd/ReadVariableOp(^simple_rnn_cell_3/MatMul/ReadVariableOp*^simple_rnn_cell_3/MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :������������������ 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::2T
(simple_rnn_cell_3/BiasAdd/ReadVariableOp(simple_rnn_cell_3/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_3/MatMul/ReadVariableOp'simple_rnn_cell_3/MatMul/ReadVariableOp2V
)simple_rnn_cell_3/MatMul_1/ReadVariableOp)simple_rnn_cell_3/MatMul_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
_
C__inference_lambda_1_layer_call_and_return_conditional_losses_99103

inputs
identityk
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
ExpandDims/dim�

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*4
_output_shapes"
 :������������������2

ExpandDimst
IdentityIdentityExpandDims:output:0*
T0*4
_output_shapes"
 :������������������2

Identity"
identityIdentity:output:0*/
_input_shapes
:������������������:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�H
�
G__inference_simple_rnn_5_layer_call_and_return_conditional_losses_99845

inputs4
0simple_rnn_cell_5_matmul_readvariableop_resource5
1simple_rnn_cell_5_biasadd_readvariableop_resource6
2simple_rnn_cell_5_matmul_1_readvariableop_resource
identity��(simple_rnn_cell_5/BiasAdd/ReadVariableOp�'simple_rnn_cell_5/MatMul/ReadVariableOp�)simple_rnn_cell_5/MatMul_1/ReadVariableOp�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
strided_slice_2�
'simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_5_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02)
'simple_rnn_cell_5/MatMul/ReadVariableOp�
simple_rnn_cell_5/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_5/MatMul�
(simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(simple_rnn_cell_5/BiasAdd/ReadVariableOp�
simple_rnn_cell_5/BiasAddBiasAdd"simple_rnn_cell_5/MatMul:product:00simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_5/BiasAdd�
)simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype02+
)simple_rnn_cell_5/MatMul_1/ReadVariableOp�
simple_rnn_cell_5/MatMul_1MatMulzeros:output:01simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_5/MatMul_1�
simple_rnn_cell_5/addAddV2"simple_rnn_cell_5/BiasAdd:output:0$simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_5/add�
simple_rnn_cell_5/TanhTanhsimple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_5/Tanh�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_5_matmul_readvariableop_resource1simple_rnn_cell_5_biasadd_readvariableop_resource2simple_rnn_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :��������� : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_99779*
condR
while_cond_99778*8
output_shapes'
%: : : : :��������� : : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
transpose_1�
IdentityIdentitystrided_slice_3:output:0)^simple_rnn_cell_5/BiasAdd/ReadVariableOp(^simple_rnn_cell_5/MatMul/ReadVariableOp*^simple_rnn_cell_5/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������ :::2T
(simple_rnn_cell_5/BiasAdd/ReadVariableOp(simple_rnn_cell_5/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_5/MatMul/ReadVariableOp'simple_rnn_cell_5/MatMul/ReadVariableOp2V
)simple_rnn_cell_5/MatMul_1/ReadVariableOp)simple_rnn_cell_5/MatMul_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�3
�
while_body_101449
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_4_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_4_matmul_readvariableop_resource;
7while_simple_rnn_cell_4_biasadd_readvariableop_resource<
8while_simple_rnn_cell_4_matmul_1_readvariableop_resource��.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp�-while/simple_rnn_cell_4/MatMul/ReadVariableOp�/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:��������� *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
-while/simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_4_matmul_readvariableop_resource_0*
_output_shapes

:  *
dtype02/
-while/simple_rnn_cell_4/MatMul/ReadVariableOp�
while/simple_rnn_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2 
while/simple_rnn_cell_4/MatMul�
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype020
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp�
while/simple_rnn_cell_4/BiasAddBiasAdd(while/simple_rnn_cell_4/MatMul:product:06while/simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2!
while/simple_rnn_cell_4/BiasAdd�
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype021
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp�
 while/simple_rnn_cell_4/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2"
 while/simple_rnn_cell_4/MatMul_1�
while/simple_rnn_cell_4/addAddV2(while/simple_rnn_cell_4/BiasAdd:output:0*while/simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2
while/simple_rnn_cell_4/add�
while/simple_rnn_cell_4/TanhTanhwhile/simple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:��������� 2
while/simple_rnn_cell_4/Tanh�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_4/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity while/simple_rnn_cell_4/Tanh:y:0/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_4_biasadd_readvariableop_resource9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_4_matmul_1_readvariableop_resource:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_4_matmul_readvariableop_resource8while_simple_rnn_cell_4_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :��������� : : :::2`
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_4/MatMul/ReadVariableOp-while/simple_rnn_cell_4/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�

�
simple_rnn_4_while_cond_1006396
2simple_rnn_4_while_simple_rnn_4_while_loop_counter<
8simple_rnn_4_while_simple_rnn_4_while_maximum_iterations"
simple_rnn_4_while_placeholder$
 simple_rnn_4_while_placeholder_1$
 simple_rnn_4_while_placeholder_28
4simple_rnn_4_while_less_simple_rnn_4_strided_slice_1N
Jsimple_rnn_4_while_simple_rnn_4_while_cond_100639___redundant_placeholder0N
Jsimple_rnn_4_while_simple_rnn_4_while_cond_100639___redundant_placeholder1N
Jsimple_rnn_4_while_simple_rnn_4_while_cond_100639___redundant_placeholder2N
Jsimple_rnn_4_while_simple_rnn_4_while_cond_100639___redundant_placeholder3
simple_rnn_4_while_identity
�
simple_rnn_4/while/LessLesssimple_rnn_4_while_placeholder4simple_rnn_4_while_less_simple_rnn_4_strided_slice_1*
T0*
_output_shapes
: 2
simple_rnn_4/while/Less�
simple_rnn_4/while/IdentityIdentitysimple_rnn_4/while/Less:z:0*
T0
*
_output_shapes
: 2
simple_rnn_4/while/Identity"C
simple_rnn_4_while_identity$simple_rnn_4/while/Identity:output:0*@
_input_shapes/
-: : : : :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�3
�
while_body_100957
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_3_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_3_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_3_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_3_matmul_readvariableop_resource;
7while_simple_rnn_cell_3_biasadd_readvariableop_resource<
8while_simple_rnn_cell_3_matmul_1_readvariableop_resource��.while/simple_rnn_cell_3/BiasAdd/ReadVariableOp�-while/simple_rnn_cell_3/MatMul/ReadVariableOp�/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
-while/simple_rnn_cell_3/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_3_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype02/
-while/simple_rnn_cell_3/MatMul/ReadVariableOp�
while/simple_rnn_cell_3/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2 
while/simple_rnn_cell_3/MatMul�
.while/simple_rnn_cell_3/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_3_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype020
.while/simple_rnn_cell_3/BiasAdd/ReadVariableOp�
while/simple_rnn_cell_3/BiasAddBiasAdd(while/simple_rnn_cell_3/MatMul:product:06while/simple_rnn_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2!
while/simple_rnn_cell_3/BiasAdd�
/while/simple_rnn_cell_3/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_3_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype021
/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp�
 while/simple_rnn_cell_3/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2"
 while/simple_rnn_cell_3/MatMul_1�
while/simple_rnn_cell_3/addAddV2(while/simple_rnn_cell_3/BiasAdd:output:0*while/simple_rnn_cell_3/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2
while/simple_rnn_cell_3/add�
while/simple_rnn_cell_3/TanhTanhwhile/simple_rnn_cell_3/add:z:0*
T0*'
_output_shapes
:��������� 2
while/simple_rnn_cell_3/Tanh�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_3/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_3/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_3/MatMul/ReadVariableOp0^while/simple_rnn_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_3/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_3/MatMul/ReadVariableOp0^while/simple_rnn_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_3/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_3/MatMul/ReadVariableOp0^while/simple_rnn_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_3/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_3/MatMul/ReadVariableOp0^while/simple_rnn_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity while/simple_rnn_cell_3/Tanh:y:0/^while/simple_rnn_cell_3/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_3/MatMul/ReadVariableOp0^while/simple_rnn_cell_3/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_3_biasadd_readvariableop_resource9while_simple_rnn_cell_3_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_3_matmul_1_readvariableop_resource:while_simple_rnn_cell_3_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_3_matmul_readvariableop_resource8while_simple_rnn_cell_3_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :��������� : : :::2`
.while/simple_rnn_cell_3/BiasAdd/ReadVariableOp.while/simple_rnn_cell_3/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_3/MatMul/ReadVariableOp-while/simple_rnn_cell_3/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�Q
�
*sequential_6_simple_rnn_5_while_body_97472P
Lsequential_6_simple_rnn_5_while_sequential_6_simple_rnn_5_while_loop_counterV
Rsequential_6_simple_rnn_5_while_sequential_6_simple_rnn_5_while_maximum_iterations/
+sequential_6_simple_rnn_5_while_placeholder1
-sequential_6_simple_rnn_5_while_placeholder_11
-sequential_6_simple_rnn_5_while_placeholder_2O
Ksequential_6_simple_rnn_5_while_sequential_6_simple_rnn_5_strided_slice_1_0�
�sequential_6_simple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_sequential_6_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0V
Rsequential_6_simple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resource_0W
Ssequential_6_simple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resource_0X
Tsequential_6_simple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0,
(sequential_6_simple_rnn_5_while_identity.
*sequential_6_simple_rnn_5_while_identity_1.
*sequential_6_simple_rnn_5_while_identity_2.
*sequential_6_simple_rnn_5_while_identity_3.
*sequential_6_simple_rnn_5_while_identity_4M
Isequential_6_simple_rnn_5_while_sequential_6_simple_rnn_5_strided_slice_1�
�sequential_6_simple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_sequential_6_simple_rnn_5_tensorarrayunstack_tensorlistfromtensorT
Psequential_6_simple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resourceU
Qsequential_6_simple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resourceV
Rsequential_6_simple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resource��Hsequential_6/simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp�Gsequential_6/simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp�Isequential_6/simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp�
Qsequential_6/simple_rnn_5/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2S
Qsequential_6/simple_rnn_5/while/TensorArrayV2Read/TensorListGetItem/element_shape�
Csequential_6/simple_rnn_5/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem�sequential_6_simple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_sequential_6_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0+sequential_6_simple_rnn_5_while_placeholderZsequential_6/simple_rnn_5/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:��������� *
element_dtype02E
Csequential_6/simple_rnn_5/while/TensorArrayV2Read/TensorListGetItem�
Gsequential_6/simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOpRsequential_6_simple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resource_0*
_output_shapes

:  *
dtype02I
Gsequential_6/simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp�
8sequential_6/simple_rnn_5/while/simple_rnn_cell_5/MatMulMatMulJsequential_6/simple_rnn_5/while/TensorArrayV2Read/TensorListGetItem:item:0Osequential_6/simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2:
8sequential_6/simple_rnn_5/while/simple_rnn_cell_5/MatMul�
Hsequential_6/simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOpSsequential_6_simple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype02J
Hsequential_6/simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp�
9sequential_6/simple_rnn_5/while/simple_rnn_cell_5/BiasAddBiasAddBsequential_6/simple_rnn_5/while/simple_rnn_cell_5/MatMul:product:0Psequential_6/simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2;
9sequential_6/simple_rnn_5/while/simple_rnn_cell_5/BiasAdd�
Isequential_6/simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOpTsequential_6_simple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype02K
Isequential_6/simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp�
:sequential_6/simple_rnn_5/while/simple_rnn_cell_5/MatMul_1MatMul-sequential_6_simple_rnn_5_while_placeholder_2Qsequential_6/simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2<
:sequential_6/simple_rnn_5/while/simple_rnn_cell_5/MatMul_1�
5sequential_6/simple_rnn_5/while/simple_rnn_cell_5/addAddV2Bsequential_6/simple_rnn_5/while/simple_rnn_cell_5/BiasAdd:output:0Dsequential_6/simple_rnn_5/while/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 27
5sequential_6/simple_rnn_5/while/simple_rnn_cell_5/add�
6sequential_6/simple_rnn_5/while/simple_rnn_cell_5/TanhTanh9sequential_6/simple_rnn_5/while/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:��������� 28
6sequential_6/simple_rnn_5/while/simple_rnn_cell_5/Tanh�
Dsequential_6/simple_rnn_5/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem-sequential_6_simple_rnn_5_while_placeholder_1+sequential_6_simple_rnn_5_while_placeholder:sequential_6/simple_rnn_5/while/simple_rnn_cell_5/Tanh:y:0*
_output_shapes
: *
element_dtype02F
Dsequential_6/simple_rnn_5/while/TensorArrayV2Write/TensorListSetItem�
%sequential_6/simple_rnn_5/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_6/simple_rnn_5/while/add/y�
#sequential_6/simple_rnn_5/while/addAddV2+sequential_6_simple_rnn_5_while_placeholder.sequential_6/simple_rnn_5/while/add/y:output:0*
T0*
_output_shapes
: 2%
#sequential_6/simple_rnn_5/while/add�
'sequential_6/simple_rnn_5/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_6/simple_rnn_5/while/add_1/y�
%sequential_6/simple_rnn_5/while/add_1AddV2Lsequential_6_simple_rnn_5_while_sequential_6_simple_rnn_5_while_loop_counter0sequential_6/simple_rnn_5/while/add_1/y:output:0*
T0*
_output_shapes
: 2'
%sequential_6/simple_rnn_5/while/add_1�
(sequential_6/simple_rnn_5/while/IdentityIdentity)sequential_6/simple_rnn_5/while/add_1:z:0I^sequential_6/simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOpH^sequential_6/simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOpJ^sequential_6/simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2*
(sequential_6/simple_rnn_5/while/Identity�
*sequential_6/simple_rnn_5/while/Identity_1IdentityRsequential_6_simple_rnn_5_while_sequential_6_simple_rnn_5_while_maximum_iterationsI^sequential_6/simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOpH^sequential_6/simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOpJ^sequential_6/simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2,
*sequential_6/simple_rnn_5/while/Identity_1�
*sequential_6/simple_rnn_5/while/Identity_2Identity'sequential_6/simple_rnn_5/while/add:z:0I^sequential_6/simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOpH^sequential_6/simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOpJ^sequential_6/simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2,
*sequential_6/simple_rnn_5/while/Identity_2�
*sequential_6/simple_rnn_5/while/Identity_3IdentityTsequential_6/simple_rnn_5/while/TensorArrayV2Write/TensorListSetItem:output_handle:0I^sequential_6/simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOpH^sequential_6/simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOpJ^sequential_6/simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2,
*sequential_6/simple_rnn_5/while/Identity_3�
*sequential_6/simple_rnn_5/while/Identity_4Identity:sequential_6/simple_rnn_5/while/simple_rnn_cell_5/Tanh:y:0I^sequential_6/simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOpH^sequential_6/simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOpJ^sequential_6/simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2,
*sequential_6/simple_rnn_5/while/Identity_4"]
(sequential_6_simple_rnn_5_while_identity1sequential_6/simple_rnn_5/while/Identity:output:0"a
*sequential_6_simple_rnn_5_while_identity_13sequential_6/simple_rnn_5/while/Identity_1:output:0"a
*sequential_6_simple_rnn_5_while_identity_23sequential_6/simple_rnn_5/while/Identity_2:output:0"a
*sequential_6_simple_rnn_5_while_identity_33sequential_6/simple_rnn_5/while/Identity_3:output:0"a
*sequential_6_simple_rnn_5_while_identity_43sequential_6/simple_rnn_5/while/Identity_4:output:0"�
Isequential_6_simple_rnn_5_while_sequential_6_simple_rnn_5_strided_slice_1Ksequential_6_simple_rnn_5_while_sequential_6_simple_rnn_5_strided_slice_1_0"�
Qsequential_6_simple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resourceSsequential_6_simple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resource_0"�
Rsequential_6_simple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resourceTsequential_6_simple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0"�
Psequential_6_simple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resourceRsequential_6_simple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resource_0"�
�sequential_6_simple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_sequential_6_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor�sequential_6_simple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_sequential_6_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :��������� : : :::2�
Hsequential_6/simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOpHsequential_6/simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp2�
Gsequential_6/simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOpGsequential_6/simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp2�
Isequential_6/simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOpIsequential_6/simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�

�
simple_rnn_5_while_cond_1007476
2simple_rnn_5_while_simple_rnn_5_while_loop_counter<
8simple_rnn_5_while_simple_rnn_5_while_maximum_iterations"
simple_rnn_5_while_placeholder$
 simple_rnn_5_while_placeholder_1$
 simple_rnn_5_while_placeholder_28
4simple_rnn_5_while_less_simple_rnn_5_strided_slice_1N
Jsimple_rnn_5_while_simple_rnn_5_while_cond_100747___redundant_placeholder0N
Jsimple_rnn_5_while_simple_rnn_5_while_cond_100747___redundant_placeholder1N
Jsimple_rnn_5_while_simple_rnn_5_while_cond_100747___redundant_placeholder2N
Jsimple_rnn_5_while_simple_rnn_5_while_cond_100747___redundant_placeholder3
simple_rnn_5_while_identity
�
simple_rnn_5/while/LessLesssimple_rnn_5_while_placeholder4simple_rnn_5_while_less_simple_rnn_5_strided_slice_1*
T0*
_output_shapes
: 2
simple_rnn_5/while/Less�
simple_rnn_5/while/IdentityIdentitysimple_rnn_5/while/Less:z:0*
T0
*
_output_shapes
: 2
simple_rnn_5/while/Identity"C
simple_rnn_5_while_identity$simple_rnn_5/while/Identity:output:0*@
_input_shapes/
-: : : : :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�
�
-__inference_simple_rnn_5_layer_call_fn_102376

inputs
unknown
	unknown_0
	unknown_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_simple_rnn_5_layer_call_and_return_conditional_losses_997332
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������ :::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�f
�
__inference__traced_save_102773
file_prefix.
*savev2_dense_22_kernel_read_readvariableop,
(savev2_dense_22_bias_read_readvariableop.
*savev2_dense_23_kernel_read_readvariableop,
(savev2_dense_23_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopD
@savev2_simple_rnn_3_simple_rnn_cell_3_kernel_read_readvariableopN
Jsavev2_simple_rnn_3_simple_rnn_cell_3_recurrent_kernel_read_readvariableopB
>savev2_simple_rnn_3_simple_rnn_cell_3_bias_read_readvariableopD
@savev2_simple_rnn_4_simple_rnn_cell_4_kernel_read_readvariableopN
Jsavev2_simple_rnn_4_simple_rnn_cell_4_recurrent_kernel_read_readvariableopB
>savev2_simple_rnn_4_simple_rnn_cell_4_bias_read_readvariableopD
@savev2_simple_rnn_5_simple_rnn_cell_5_kernel_read_readvariableopN
Jsavev2_simple_rnn_5_simple_rnn_cell_5_recurrent_kernel_read_readvariableopB
>savev2_simple_rnn_5_simple_rnn_cell_5_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_22_kernel_m_read_readvariableop3
/savev2_adam_dense_22_bias_m_read_readvariableop5
1savev2_adam_dense_23_kernel_m_read_readvariableop3
/savev2_adam_dense_23_bias_m_read_readvariableopK
Gsavev2_adam_simple_rnn_3_simple_rnn_cell_3_kernel_m_read_readvariableopU
Qsavev2_adam_simple_rnn_3_simple_rnn_cell_3_recurrent_kernel_m_read_readvariableopI
Esavev2_adam_simple_rnn_3_simple_rnn_cell_3_bias_m_read_readvariableopK
Gsavev2_adam_simple_rnn_4_simple_rnn_cell_4_kernel_m_read_readvariableopU
Qsavev2_adam_simple_rnn_4_simple_rnn_cell_4_recurrent_kernel_m_read_readvariableopI
Esavev2_adam_simple_rnn_4_simple_rnn_cell_4_bias_m_read_readvariableopK
Gsavev2_adam_simple_rnn_5_simple_rnn_cell_5_kernel_m_read_readvariableopU
Qsavev2_adam_simple_rnn_5_simple_rnn_cell_5_recurrent_kernel_m_read_readvariableopI
Esavev2_adam_simple_rnn_5_simple_rnn_cell_5_bias_m_read_readvariableop5
1savev2_adam_dense_22_kernel_v_read_readvariableop3
/savev2_adam_dense_22_bias_v_read_readvariableop5
1savev2_adam_dense_23_kernel_v_read_readvariableop3
/savev2_adam_dense_23_bias_v_read_readvariableopK
Gsavev2_adam_simple_rnn_3_simple_rnn_cell_3_kernel_v_read_readvariableopU
Qsavev2_adam_simple_rnn_3_simple_rnn_cell_3_recurrent_kernel_v_read_readvariableopI
Esavev2_adam_simple_rnn_3_simple_rnn_cell_3_bias_v_read_readvariableopK
Gsavev2_adam_simple_rnn_4_simple_rnn_cell_4_kernel_v_read_readvariableopU
Qsavev2_adam_simple_rnn_4_simple_rnn_cell_4_recurrent_kernel_v_read_readvariableopI
Esavev2_adam_simple_rnn_4_simple_rnn_cell_4_bias_v_read_readvariableopK
Gsavev2_adam_simple_rnn_5_simple_rnn_cell_5_kernel_v_read_readvariableopU
Qsavev2_adam_simple_rnn_5_simple_rnn_cell_5_recurrent_kernel_v_read_readvariableopI
Esavev2_adam_simple_rnn_5_simple_rnn_cell_5_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*�
value�B�/B6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*q
valuehBf/B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_22_kernel_read_readvariableop(savev2_dense_22_bias_read_readvariableop*savev2_dense_23_kernel_read_readvariableop(savev2_dense_23_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop@savev2_simple_rnn_3_simple_rnn_cell_3_kernel_read_readvariableopJsavev2_simple_rnn_3_simple_rnn_cell_3_recurrent_kernel_read_readvariableop>savev2_simple_rnn_3_simple_rnn_cell_3_bias_read_readvariableop@savev2_simple_rnn_4_simple_rnn_cell_4_kernel_read_readvariableopJsavev2_simple_rnn_4_simple_rnn_cell_4_recurrent_kernel_read_readvariableop>savev2_simple_rnn_4_simple_rnn_cell_4_bias_read_readvariableop@savev2_simple_rnn_5_simple_rnn_cell_5_kernel_read_readvariableopJsavev2_simple_rnn_5_simple_rnn_cell_5_recurrent_kernel_read_readvariableop>savev2_simple_rnn_5_simple_rnn_cell_5_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_22_kernel_m_read_readvariableop/savev2_adam_dense_22_bias_m_read_readvariableop1savev2_adam_dense_23_kernel_m_read_readvariableop/savev2_adam_dense_23_bias_m_read_readvariableopGsavev2_adam_simple_rnn_3_simple_rnn_cell_3_kernel_m_read_readvariableopQsavev2_adam_simple_rnn_3_simple_rnn_cell_3_recurrent_kernel_m_read_readvariableopEsavev2_adam_simple_rnn_3_simple_rnn_cell_3_bias_m_read_readvariableopGsavev2_adam_simple_rnn_4_simple_rnn_cell_4_kernel_m_read_readvariableopQsavev2_adam_simple_rnn_4_simple_rnn_cell_4_recurrent_kernel_m_read_readvariableopEsavev2_adam_simple_rnn_4_simple_rnn_cell_4_bias_m_read_readvariableopGsavev2_adam_simple_rnn_5_simple_rnn_cell_5_kernel_m_read_readvariableopQsavev2_adam_simple_rnn_5_simple_rnn_cell_5_recurrent_kernel_m_read_readvariableopEsavev2_adam_simple_rnn_5_simple_rnn_cell_5_bias_m_read_readvariableop1savev2_adam_dense_22_kernel_v_read_readvariableop/savev2_adam_dense_22_bias_v_read_readvariableop1savev2_adam_dense_23_kernel_v_read_readvariableop/savev2_adam_dense_23_bias_v_read_readvariableopGsavev2_adam_simple_rnn_3_simple_rnn_cell_3_kernel_v_read_readvariableopQsavev2_adam_simple_rnn_3_simple_rnn_cell_3_recurrent_kernel_v_read_readvariableopEsavev2_adam_simple_rnn_3_simple_rnn_cell_3_bias_v_read_readvariableopGsavev2_adam_simple_rnn_4_simple_rnn_cell_4_kernel_v_read_readvariableopQsavev2_adam_simple_rnn_4_simple_rnn_cell_4_recurrent_kernel_v_read_readvariableopEsavev2_adam_simple_rnn_4_simple_rnn_cell_4_bias_v_read_readvariableopGsavev2_adam_simple_rnn_5_simple_rnn_cell_5_kernel_v_read_readvariableopQsavev2_adam_simple_rnn_5_simple_rnn_cell_5_recurrent_kernel_v_read_readvariableopEsavev2_adam_simple_rnn_5_simple_rnn_cell_5_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *=
dtypes3
12/	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: : :::: : : : : : :  : :  :  : :  :  : : : : :::: :  : :  :  : :  :  : : :::: :  : :  :  : :  :  : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :$
 

_output_shapes

: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

:  :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

:  :$ 

_output_shapes

:  : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

:  :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

:  :$  

_output_shapes

:  : !

_output_shapes
: :$" 

_output_shapes

: : #

_output_shapes
::$$ 

_output_shapes

:: %

_output_shapes
::$& 

_output_shapes

: :$' 

_output_shapes

:  : (

_output_shapes
: :$) 

_output_shapes

:  :$* 

_output_shapes

:  : +

_output_shapes
: :$, 

_output_shapes

:  :$- 

_output_shapes

:  : .

_output_shapes
: :/

_output_shapes
: 
�#
�
while_body_98386
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0#
while_simple_rnn_cell_4_98408_0#
while_simple_rnn_cell_4_98410_0#
while_simple_rnn_cell_4_98412_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor!
while_simple_rnn_cell_4_98408!
while_simple_rnn_cell_4_98410!
while_simple_rnn_cell_4_98412��/while/simple_rnn_cell_4/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:��������� *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
/while/simple_rnn_cell_4/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_simple_rnn_cell_4_98408_0while_simple_rnn_cell_4_98410_0while_simple_rnn_cell_4_98412_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:��������� :��������� *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_9811221
/while/simple_rnn_cell_4/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder8while/simple_rnn_cell_4/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:00^while/simple_rnn_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations0^while/simple_rnn_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:00^while/simple_rnn_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/simple_rnn_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity8while/simple_rnn_cell_4/StatefulPartitionedCall:output:10^while/simple_rnn_cell_4/StatefulPartitionedCall*
T0*'
_output_shapes
:��������� 2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"@
while_simple_rnn_cell_4_98408while_simple_rnn_cell_4_98408_0"@
while_simple_rnn_cell_4_98410while_simple_rnn_cell_4_98410_0"@
while_simple_rnn_cell_4_98412while_simple_rnn_cell_4_98412_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :��������� : : :::2b
/while/simple_rnn_cell_4/StatefulPartitionedCall/while/simple_rnn_cell_4/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_99160
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_99160___redundant_placeholder03
/while_while_cond_99160___redundant_placeholder13
/while_while_cond_99160___redundant_placeholder23
/while_while_cond_99160___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�
�
while_cond_99525
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_99525___redundant_placeholder03
/while_while_cond_99525___redundant_placeholder13
/while_while_cond_99525___redundant_placeholder23
/while_while_cond_99525___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�H
�
H__inference_simple_rnn_3_layer_call_and_return_conditional_losses_101135
inputs_04
0simple_rnn_cell_3_matmul_readvariableop_resource5
1simple_rnn_cell_3_biasadd_readvariableop_resource6
2simple_rnn_cell_3_matmul_1_readvariableop_resource
identity��(simple_rnn_cell_3/BiasAdd/ReadVariableOp�'simple_rnn_cell_3/MatMul/ReadVariableOp�)simple_rnn_cell_3/MatMul_1/ReadVariableOp�whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
'simple_rnn_cell_3/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02)
'simple_rnn_cell_3/MatMul/ReadVariableOp�
simple_rnn_cell_3/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_3/MatMul�
(simple_rnn_cell_3/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(simple_rnn_cell_3/BiasAdd/ReadVariableOp�
simple_rnn_cell_3/BiasAddBiasAdd"simple_rnn_cell_3/MatMul:product:00simple_rnn_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_3/BiasAdd�
)simple_rnn_cell_3/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_3_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype02+
)simple_rnn_cell_3/MatMul_1/ReadVariableOp�
simple_rnn_cell_3/MatMul_1MatMulzeros:output:01simple_rnn_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_3/MatMul_1�
simple_rnn_cell_3/addAddV2"simple_rnn_cell_3/BiasAdd:output:0$simple_rnn_cell_3/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_3/add�
simple_rnn_cell_3/TanhTanhsimple_rnn_cell_3/add:z:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_3/Tanh�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_3_matmul_readvariableop_resource1simple_rnn_cell_3_biasadd_readvariableop_resource2simple_rnn_cell_3_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :��������� : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_101069*
condR
while_cond_101068*8
output_shapes'
%: : : : :��������� : : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
transpose_1�
IdentityIdentitytranspose_1:y:0)^simple_rnn_cell_3/BiasAdd/ReadVariableOp(^simple_rnn_cell_3/MatMul/ReadVariableOp*^simple_rnn_cell_3/MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :������������������ 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::2T
(simple_rnn_cell_3/BiasAdd/ReadVariableOp(simple_rnn_cell_3/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_3/MatMul/ReadVariableOp'simple_rnn_cell_3/MatMul/ReadVariableOp2V
)simple_rnn_cell_3/MatMul_1/ReadVariableOp)simple_rnn_cell_3/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�
E
)__inference_lambda_1_layer_call_fn_100906

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_lambda_1_layer_call_and_return_conditional_losses_990972
PartitionedCally
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :������������������2

Identity"
identityIdentity:output:0*/
_input_shapes
:������������������:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
�
while_cond_101448
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_101448___redundant_placeholder04
0while_while_cond_101448___redundant_placeholder14
0while_while_cond_101448___redundant_placeholder24
0while_while_cond_101448___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�
�
-__inference_simple_rnn_5_layer_call_fn_102387

inputs
unknown
	unknown_0
	unknown_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_simple_rnn_5_layer_call_and_return_conditional_losses_998452
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������ :::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�3
�
while_body_102299
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_5_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_5_matmul_readvariableop_resource;
7while_simple_rnn_cell_5_biasadd_readvariableop_resource<
8while_simple_rnn_cell_5_matmul_1_readvariableop_resource��.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp�-while/simple_rnn_cell_5/MatMul/ReadVariableOp�/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:��������� *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
-while/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_5_matmul_readvariableop_resource_0*
_output_shapes

:  *
dtype02/
-while/simple_rnn_cell_5/MatMul/ReadVariableOp�
while/simple_rnn_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2 
while/simple_rnn_cell_5/MatMul�
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype020
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp�
while/simple_rnn_cell_5/BiasAddBiasAdd(while/simple_rnn_cell_5/MatMul:product:06while/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2!
while/simple_rnn_cell_5/BiasAdd�
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype021
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp�
 while/simple_rnn_cell_5/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2"
 while/simple_rnn_cell_5/MatMul_1�
while/simple_rnn_cell_5/addAddV2(while/simple_rnn_cell_5/BiasAdd:output:0*while/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2
while/simple_rnn_cell_5/add�
while/simple_rnn_cell_5/TanhTanhwhile/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:��������� 2
while/simple_rnn_cell_5/Tanh�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_5/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity while/simple_rnn_cell_5/Tanh:y:0/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_5_biasadd_readvariableop_resource9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_5_matmul_1_readvariableop_resource:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_5_matmul_readvariableop_resource8while_simple_rnn_cell_5_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :��������� : : :::2`
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_5/MatMul/ReadVariableOp-while/simple_rnn_cell_5/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�
�
M__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_102505

inputs
states_0"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2	
BiasAdd�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2

MatMul_1k
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2
addO
TanhTanhadd:z:0*
T0*'
_output_shapes
:��������� 2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2

Identity�

Identity_1IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:��������� :��������� :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states/0
�3
�
while_body_101941
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_5_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_5_matmul_readvariableop_resource;
7while_simple_rnn_cell_5_biasadd_readvariableop_resource<
8while_simple_rnn_cell_5_matmul_1_readvariableop_resource��.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp�-while/simple_rnn_cell_5/MatMul/ReadVariableOp�/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:��������� *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
-while/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_5_matmul_readvariableop_resource_0*
_output_shapes

:  *
dtype02/
-while/simple_rnn_cell_5/MatMul/ReadVariableOp�
while/simple_rnn_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2 
while/simple_rnn_cell_5/MatMul�
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype020
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp�
while/simple_rnn_cell_5/BiasAddBiasAdd(while/simple_rnn_cell_5/MatMul:product:06while/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2!
while/simple_rnn_cell_5/BiasAdd�
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype021
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp�
 while/simple_rnn_cell_5/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2"
 while/simple_rnn_cell_5/MatMul_1�
while/simple_rnn_cell_5/addAddV2(while/simple_rnn_cell_5/BiasAdd:output:0*while/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2
while/simple_rnn_cell_5/add�
while/simple_rnn_cell_5/TanhTanhwhile/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:��������� 2
while/simple_rnn_cell_5/Tanh�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_5/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity while/simple_rnn_cell_5/Tanh:y:0/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_5_biasadd_readvariableop_resource9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_5_matmul_1_readvariableop_resource:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_5_matmul_readvariableop_resource8while_simple_rnn_cell_5_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :��������� : : :::2`
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_5/MatMul/ReadVariableOp-while/simple_rnn_cell_5/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�H
�
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_99339

inputs4
0simple_rnn_cell_3_matmul_readvariableop_resource5
1simple_rnn_cell_3_biasadd_readvariableop_resource6
2simple_rnn_cell_3_matmul_1_readvariableop_resource
identity��(simple_rnn_cell_3/BiasAdd/ReadVariableOp�'simple_rnn_cell_3/MatMul/ReadVariableOp�)simple_rnn_cell_3/MatMul_1/ReadVariableOp�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
'simple_rnn_cell_3/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02)
'simple_rnn_cell_3/MatMul/ReadVariableOp�
simple_rnn_cell_3/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_3/MatMul�
(simple_rnn_cell_3/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(simple_rnn_cell_3/BiasAdd/ReadVariableOp�
simple_rnn_cell_3/BiasAddBiasAdd"simple_rnn_cell_3/MatMul:product:00simple_rnn_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_3/BiasAdd�
)simple_rnn_cell_3/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_3_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype02+
)simple_rnn_cell_3/MatMul_1/ReadVariableOp�
simple_rnn_cell_3/MatMul_1MatMulzeros:output:01simple_rnn_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_3/MatMul_1�
simple_rnn_cell_3/addAddV2"simple_rnn_cell_3/BiasAdd:output:0$simple_rnn_cell_3/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_3/add�
simple_rnn_cell_3/TanhTanhsimple_rnn_cell_3/add:z:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_3/Tanh�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_3_matmul_readvariableop_resource1simple_rnn_cell_3_biasadd_readvariableop_resource2simple_rnn_cell_3_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :��������� : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_99273*
condR
while_cond_99272*8
output_shapes'
%: : : : :��������� : : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
transpose_1�
IdentityIdentitytranspose_1:y:0)^simple_rnn_cell_3/BiasAdd/ReadVariableOp(^simple_rnn_cell_3/MatMul/ReadVariableOp*^simple_rnn_cell_3/MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :������������������ 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::2T
(simple_rnn_cell_3/BiasAdd/ReadVariableOp(simple_rnn_cell_3/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_3/MatMul/ReadVariableOp'simple_rnn_cell_3/MatMul/ReadVariableOp2V
)simple_rnn_cell_3/MatMul_1/ReadVariableOp)simple_rnn_cell_3/MatMul_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�3
�
while_body_101695
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_4_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_4_matmul_readvariableop_resource;
7while_simple_rnn_cell_4_biasadd_readvariableop_resource<
8while_simple_rnn_cell_4_matmul_1_readvariableop_resource��.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp�-while/simple_rnn_cell_4/MatMul/ReadVariableOp�/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:��������� *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
-while/simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_4_matmul_readvariableop_resource_0*
_output_shapes

:  *
dtype02/
-while/simple_rnn_cell_4/MatMul/ReadVariableOp�
while/simple_rnn_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2 
while/simple_rnn_cell_4/MatMul�
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype020
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp�
while/simple_rnn_cell_4/BiasAddBiasAdd(while/simple_rnn_cell_4/MatMul:product:06while/simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2!
while/simple_rnn_cell_4/BiasAdd�
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype021
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp�
 while/simple_rnn_cell_4/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2"
 while/simple_rnn_cell_4/MatMul_1�
while/simple_rnn_cell_4/addAddV2(while/simple_rnn_cell_4/BiasAdd:output:0*while/simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2
while/simple_rnn_cell_4/add�
while/simple_rnn_cell_4/TanhTanhwhile/simple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:��������� 2
while/simple_rnn_cell_4/Tanh�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_4/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity while/simple_rnn_cell_4/Tanh:y:0/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_4_biasadd_readvariableop_resource9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_4_matmul_1_readvariableop_resource:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_4_matmul_readvariableop_resource8while_simple_rnn_cell_4_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :��������� : : :::2`
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_4/MatMul/ReadVariableOp-while/simple_rnn_cell_4/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�H
�
H__inference_simple_rnn_3_layer_call_and_return_conditional_losses_101023
inputs_04
0simple_rnn_cell_3_matmul_readvariableop_resource5
1simple_rnn_cell_3_biasadd_readvariableop_resource6
2simple_rnn_cell_3_matmul_1_readvariableop_resource
identity��(simple_rnn_cell_3/BiasAdd/ReadVariableOp�'simple_rnn_cell_3/MatMul/ReadVariableOp�)simple_rnn_cell_3/MatMul_1/ReadVariableOp�whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
'simple_rnn_cell_3/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02)
'simple_rnn_cell_3/MatMul/ReadVariableOp�
simple_rnn_cell_3/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_3/MatMul�
(simple_rnn_cell_3/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(simple_rnn_cell_3/BiasAdd/ReadVariableOp�
simple_rnn_cell_3/BiasAddBiasAdd"simple_rnn_cell_3/MatMul:product:00simple_rnn_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_3/BiasAdd�
)simple_rnn_cell_3/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_3_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype02+
)simple_rnn_cell_3/MatMul_1/ReadVariableOp�
simple_rnn_cell_3/MatMul_1MatMulzeros:output:01simple_rnn_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_3/MatMul_1�
simple_rnn_cell_3/addAddV2"simple_rnn_cell_3/BiasAdd:output:0$simple_rnn_cell_3/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_3/add�
simple_rnn_cell_3/TanhTanhsimple_rnn_cell_3/add:z:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_3/Tanh�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_3_matmul_readvariableop_resource1simple_rnn_cell_3_biasadd_readvariableop_resource2simple_rnn_cell_3_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :��������� : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_100957*
condR
while_cond_100956*8
output_shapes'
%: : : : :��������� : : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
transpose_1�
IdentityIdentitytranspose_1:y:0)^simple_rnn_cell_3/BiasAdd/ReadVariableOp(^simple_rnn_cell_3/MatMul/ReadVariableOp*^simple_rnn_cell_3/MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :������������������ 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::2T
(simple_rnn_cell_3/BiasAdd/ReadVariableOp(simple_rnn_cell_3/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_3/MatMul/ReadVariableOp'simple_rnn_cell_3/MatMul/ReadVariableOp2V
)simple_rnn_cell_3/MatMul_1/ReadVariableOp)simple_rnn_cell_3/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
؝
�
H__inference_sequential_6_layer_call_and_return_conditional_losses_100827

inputsA
=simple_rnn_3_simple_rnn_cell_3_matmul_readvariableop_resourceB
>simple_rnn_3_simple_rnn_cell_3_biasadd_readvariableop_resourceC
?simple_rnn_3_simple_rnn_cell_3_matmul_1_readvariableop_resourceA
=simple_rnn_4_simple_rnn_cell_4_matmul_readvariableop_resourceB
>simple_rnn_4_simple_rnn_cell_4_biasadd_readvariableop_resourceC
?simple_rnn_4_simple_rnn_cell_4_matmul_1_readvariableop_resourceA
=simple_rnn_5_simple_rnn_cell_5_matmul_readvariableop_resourceB
>simple_rnn_5_simple_rnn_cell_5_biasadd_readvariableop_resourceC
?simple_rnn_5_simple_rnn_cell_5_matmul_1_readvariableop_resource+
'dense_22_matmul_readvariableop_resource,
(dense_22_biasadd_readvariableop_resource+
'dense_23_matmul_readvariableop_resource,
(dense_23_biasadd_readvariableop_resource
identity��dense_22/BiasAdd/ReadVariableOp�dense_22/MatMul/ReadVariableOp�dense_23/BiasAdd/ReadVariableOp�dense_23/MatMul/ReadVariableOp�5simple_rnn_3/simple_rnn_cell_3/BiasAdd/ReadVariableOp�4simple_rnn_3/simple_rnn_cell_3/MatMul/ReadVariableOp�6simple_rnn_3/simple_rnn_cell_3/MatMul_1/ReadVariableOp�simple_rnn_3/while�5simple_rnn_4/simple_rnn_cell_4/BiasAdd/ReadVariableOp�4simple_rnn_4/simple_rnn_cell_4/MatMul/ReadVariableOp�6simple_rnn_4/simple_rnn_cell_4/MatMul_1/ReadVariableOp�simple_rnn_4/while�5simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOp�4simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOp�6simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp�simple_rnn_5/while}
lambda_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
lambda_1/ExpandDims/dim�
lambda_1/ExpandDims
ExpandDimsinputs lambda_1/ExpandDims/dim:output:0*
T0*4
_output_shapes"
 :������������������2
lambda_1/ExpandDimst
simple_rnn_3/ShapeShapelambda_1/ExpandDims:output:0*
T0*
_output_shapes
:2
simple_rnn_3/Shape�
 simple_rnn_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 simple_rnn_3/strided_slice/stack�
"simple_rnn_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_3/strided_slice/stack_1�
"simple_rnn_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_3/strided_slice/stack_2�
simple_rnn_3/strided_sliceStridedSlicesimple_rnn_3/Shape:output:0)simple_rnn_3/strided_slice/stack:output:0+simple_rnn_3/strided_slice/stack_1:output:0+simple_rnn_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_3/strided_slicev
simple_rnn_3/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn_3/zeros/mul/y�
simple_rnn_3/zeros/mulMul#simple_rnn_3/strided_slice:output:0!simple_rnn_3/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_3/zeros/muly
simple_rnn_3/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
simple_rnn_3/zeros/Less/y�
simple_rnn_3/zeros/LessLesssimple_rnn_3/zeros/mul:z:0"simple_rnn_3/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_3/zeros/Less|
simple_rnn_3/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn_3/zeros/packed/1�
simple_rnn_3/zeros/packedPack#simple_rnn_3/strided_slice:output:0$simple_rnn_3/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
simple_rnn_3/zeros/packedy
simple_rnn_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
simple_rnn_3/zeros/Const�
simple_rnn_3/zerosFill"simple_rnn_3/zeros/packed:output:0!simple_rnn_3/zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_3/zeros�
simple_rnn_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_3/transpose/perm�
simple_rnn_3/transpose	Transposelambda_1/ExpandDims:output:0$simple_rnn_3/transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
simple_rnn_3/transposev
simple_rnn_3/Shape_1Shapesimple_rnn_3/transpose:y:0*
T0*
_output_shapes
:2
simple_rnn_3/Shape_1�
"simple_rnn_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn_3/strided_slice_1/stack�
$simple_rnn_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_3/strided_slice_1/stack_1�
$simple_rnn_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_3/strided_slice_1/stack_2�
simple_rnn_3/strided_slice_1StridedSlicesimple_rnn_3/Shape_1:output:0+simple_rnn_3/strided_slice_1/stack:output:0-simple_rnn_3/strided_slice_1/stack_1:output:0-simple_rnn_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_3/strided_slice_1�
(simple_rnn_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2*
(simple_rnn_3/TensorArrayV2/element_shape�
simple_rnn_3/TensorArrayV2TensorListReserve1simple_rnn_3/TensorArrayV2/element_shape:output:0%simple_rnn_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_3/TensorArrayV2�
Bsimple_rnn_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2D
Bsimple_rnn_3/TensorArrayUnstack/TensorListFromTensor/element_shape�
4simple_rnn_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_3/transpose:y:0Ksimple_rnn_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type026
4simple_rnn_3/TensorArrayUnstack/TensorListFromTensor�
"simple_rnn_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn_3/strided_slice_2/stack�
$simple_rnn_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_3/strided_slice_2/stack_1�
$simple_rnn_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_3/strided_slice_2/stack_2�
simple_rnn_3/strided_slice_2StridedSlicesimple_rnn_3/transpose:y:0+simple_rnn_3/strided_slice_2/stack:output:0-simple_rnn_3/strided_slice_2/stack_1:output:0-simple_rnn_3/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
simple_rnn_3/strided_slice_2�
4simple_rnn_3/simple_rnn_cell_3/MatMul/ReadVariableOpReadVariableOp=simple_rnn_3_simple_rnn_cell_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype026
4simple_rnn_3/simple_rnn_cell_3/MatMul/ReadVariableOp�
%simple_rnn_3/simple_rnn_cell_3/MatMulMatMul%simple_rnn_3/strided_slice_2:output:0<simple_rnn_3/simple_rnn_cell_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2'
%simple_rnn_3/simple_rnn_cell_3/MatMul�
5simple_rnn_3/simple_rnn_cell_3/BiasAdd/ReadVariableOpReadVariableOp>simple_rnn_3_simple_rnn_cell_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype027
5simple_rnn_3/simple_rnn_cell_3/BiasAdd/ReadVariableOp�
&simple_rnn_3/simple_rnn_cell_3/BiasAddBiasAdd/simple_rnn_3/simple_rnn_cell_3/MatMul:product:0=simple_rnn_3/simple_rnn_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2(
&simple_rnn_3/simple_rnn_cell_3/BiasAdd�
6simple_rnn_3/simple_rnn_cell_3/MatMul_1/ReadVariableOpReadVariableOp?simple_rnn_3_simple_rnn_cell_3_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype028
6simple_rnn_3/simple_rnn_cell_3/MatMul_1/ReadVariableOp�
'simple_rnn_3/simple_rnn_cell_3/MatMul_1MatMulsimple_rnn_3/zeros:output:0>simple_rnn_3/simple_rnn_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2)
'simple_rnn_3/simple_rnn_cell_3/MatMul_1�
"simple_rnn_3/simple_rnn_cell_3/addAddV2/simple_rnn_3/simple_rnn_cell_3/BiasAdd:output:01simple_rnn_3/simple_rnn_cell_3/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2$
"simple_rnn_3/simple_rnn_cell_3/add�
#simple_rnn_3/simple_rnn_cell_3/TanhTanh&simple_rnn_3/simple_rnn_cell_3/add:z:0*
T0*'
_output_shapes
:��������� 2%
#simple_rnn_3/simple_rnn_cell_3/Tanh�
*simple_rnn_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2,
*simple_rnn_3/TensorArrayV2_1/element_shape�
simple_rnn_3/TensorArrayV2_1TensorListReserve3simple_rnn_3/TensorArrayV2_1/element_shape:output:0%simple_rnn_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_3/TensorArrayV2_1h
simple_rnn_3/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn_3/time�
%simple_rnn_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%simple_rnn_3/while/maximum_iterations�
simple_rnn_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2!
simple_rnn_3/while/loop_counter�
simple_rnn_3/whileWhile(simple_rnn_3/while/loop_counter:output:0.simple_rnn_3/while/maximum_iterations:output:0simple_rnn_3/time:output:0%simple_rnn_3/TensorArrayV2_1:handle:0simple_rnn_3/zeros:output:0%simple_rnn_3/strided_slice_1:output:0Dsimple_rnn_3/TensorArrayUnstack/TensorListFromTensor:output_handle:0=simple_rnn_3_simple_rnn_cell_3_matmul_readvariableop_resource>simple_rnn_3_simple_rnn_cell_3_biasadd_readvariableop_resource?simple_rnn_3_simple_rnn_cell_3_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :��������� : : : : : *%
_read_only_resource_inputs
	**
body"R 
simple_rnn_3_while_body_100532**
cond"R 
simple_rnn_3_while_cond_100531*8
output_shapes'
%: : : : :��������� : : : : : *
parallel_iterations 2
simple_rnn_3/while�
=simple_rnn_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2?
=simple_rnn_3/TensorArrayV2Stack/TensorListStack/element_shape�
/simple_rnn_3/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_3/while:output:3Fsimple_rnn_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������ *
element_dtype021
/simple_rnn_3/TensorArrayV2Stack/TensorListStack�
"simple_rnn_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2$
"simple_rnn_3/strided_slice_3/stack�
$simple_rnn_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$simple_rnn_3/strided_slice_3/stack_1�
$simple_rnn_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_3/strided_slice_3/stack_2�
simple_rnn_3/strided_slice_3StridedSlice8simple_rnn_3/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_3/strided_slice_3/stack:output:0-simple_rnn_3/strided_slice_3/stack_1:output:0-simple_rnn_3/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
simple_rnn_3/strided_slice_3�
simple_rnn_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_3/transpose_1/perm�
simple_rnn_3/transpose_1	Transpose8simple_rnn_3/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_3/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
simple_rnn_3/transpose_1t
simple_rnn_4/ShapeShapesimple_rnn_3/transpose_1:y:0*
T0*
_output_shapes
:2
simple_rnn_4/Shape�
 simple_rnn_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 simple_rnn_4/strided_slice/stack�
"simple_rnn_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_4/strided_slice/stack_1�
"simple_rnn_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_4/strided_slice/stack_2�
simple_rnn_4/strided_sliceStridedSlicesimple_rnn_4/Shape:output:0)simple_rnn_4/strided_slice/stack:output:0+simple_rnn_4/strided_slice/stack_1:output:0+simple_rnn_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_4/strided_slicev
simple_rnn_4/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn_4/zeros/mul/y�
simple_rnn_4/zeros/mulMul#simple_rnn_4/strided_slice:output:0!simple_rnn_4/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_4/zeros/muly
simple_rnn_4/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
simple_rnn_4/zeros/Less/y�
simple_rnn_4/zeros/LessLesssimple_rnn_4/zeros/mul:z:0"simple_rnn_4/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_4/zeros/Less|
simple_rnn_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn_4/zeros/packed/1�
simple_rnn_4/zeros/packedPack#simple_rnn_4/strided_slice:output:0$simple_rnn_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
simple_rnn_4/zeros/packedy
simple_rnn_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
simple_rnn_4/zeros/Const�
simple_rnn_4/zerosFill"simple_rnn_4/zeros/packed:output:0!simple_rnn_4/zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_4/zeros�
simple_rnn_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_4/transpose/perm�
simple_rnn_4/transpose	Transposesimple_rnn_3/transpose_1:y:0$simple_rnn_4/transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
simple_rnn_4/transposev
simple_rnn_4/Shape_1Shapesimple_rnn_4/transpose:y:0*
T0*
_output_shapes
:2
simple_rnn_4/Shape_1�
"simple_rnn_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn_4/strided_slice_1/stack�
$simple_rnn_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_4/strided_slice_1/stack_1�
$simple_rnn_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_4/strided_slice_1/stack_2�
simple_rnn_4/strided_slice_1StridedSlicesimple_rnn_4/Shape_1:output:0+simple_rnn_4/strided_slice_1/stack:output:0-simple_rnn_4/strided_slice_1/stack_1:output:0-simple_rnn_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_4/strided_slice_1�
(simple_rnn_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2*
(simple_rnn_4/TensorArrayV2/element_shape�
simple_rnn_4/TensorArrayV2TensorListReserve1simple_rnn_4/TensorArrayV2/element_shape:output:0%simple_rnn_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_4/TensorArrayV2�
Bsimple_rnn_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2D
Bsimple_rnn_4/TensorArrayUnstack/TensorListFromTensor/element_shape�
4simple_rnn_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_4/transpose:y:0Ksimple_rnn_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type026
4simple_rnn_4/TensorArrayUnstack/TensorListFromTensor�
"simple_rnn_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn_4/strided_slice_2/stack�
$simple_rnn_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_4/strided_slice_2/stack_1�
$simple_rnn_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_4/strided_slice_2/stack_2�
simple_rnn_4/strided_slice_2StridedSlicesimple_rnn_4/transpose:y:0+simple_rnn_4/strided_slice_2/stack:output:0-simple_rnn_4/strided_slice_2/stack_1:output:0-simple_rnn_4/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
simple_rnn_4/strided_slice_2�
4simple_rnn_4/simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp=simple_rnn_4_simple_rnn_cell_4_matmul_readvariableop_resource*
_output_shapes

:  *
dtype026
4simple_rnn_4/simple_rnn_cell_4/MatMul/ReadVariableOp�
%simple_rnn_4/simple_rnn_cell_4/MatMulMatMul%simple_rnn_4/strided_slice_2:output:0<simple_rnn_4/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2'
%simple_rnn_4/simple_rnn_cell_4/MatMul�
5simple_rnn_4/simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp>simple_rnn_4_simple_rnn_cell_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype027
5simple_rnn_4/simple_rnn_cell_4/BiasAdd/ReadVariableOp�
&simple_rnn_4/simple_rnn_cell_4/BiasAddBiasAdd/simple_rnn_4/simple_rnn_cell_4/MatMul:product:0=simple_rnn_4/simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2(
&simple_rnn_4/simple_rnn_cell_4/BiasAdd�
6simple_rnn_4/simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp?simple_rnn_4_simple_rnn_cell_4_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype028
6simple_rnn_4/simple_rnn_cell_4/MatMul_1/ReadVariableOp�
'simple_rnn_4/simple_rnn_cell_4/MatMul_1MatMulsimple_rnn_4/zeros:output:0>simple_rnn_4/simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2)
'simple_rnn_4/simple_rnn_cell_4/MatMul_1�
"simple_rnn_4/simple_rnn_cell_4/addAddV2/simple_rnn_4/simple_rnn_cell_4/BiasAdd:output:01simple_rnn_4/simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2$
"simple_rnn_4/simple_rnn_cell_4/add�
#simple_rnn_4/simple_rnn_cell_4/TanhTanh&simple_rnn_4/simple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:��������� 2%
#simple_rnn_4/simple_rnn_cell_4/Tanh�
*simple_rnn_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2,
*simple_rnn_4/TensorArrayV2_1/element_shape�
simple_rnn_4/TensorArrayV2_1TensorListReserve3simple_rnn_4/TensorArrayV2_1/element_shape:output:0%simple_rnn_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_4/TensorArrayV2_1h
simple_rnn_4/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn_4/time�
%simple_rnn_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%simple_rnn_4/while/maximum_iterations�
simple_rnn_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2!
simple_rnn_4/while/loop_counter�
simple_rnn_4/whileWhile(simple_rnn_4/while/loop_counter:output:0.simple_rnn_4/while/maximum_iterations:output:0simple_rnn_4/time:output:0%simple_rnn_4/TensorArrayV2_1:handle:0simple_rnn_4/zeros:output:0%simple_rnn_4/strided_slice_1:output:0Dsimple_rnn_4/TensorArrayUnstack/TensorListFromTensor:output_handle:0=simple_rnn_4_simple_rnn_cell_4_matmul_readvariableop_resource>simple_rnn_4_simple_rnn_cell_4_biasadd_readvariableop_resource?simple_rnn_4_simple_rnn_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :��������� : : : : : *%
_read_only_resource_inputs
	**
body"R 
simple_rnn_4_while_body_100640**
cond"R 
simple_rnn_4_while_cond_100639*8
output_shapes'
%: : : : :��������� : : : : : *
parallel_iterations 2
simple_rnn_4/while�
=simple_rnn_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2?
=simple_rnn_4/TensorArrayV2Stack/TensorListStack/element_shape�
/simple_rnn_4/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_4/while:output:3Fsimple_rnn_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������ *
element_dtype021
/simple_rnn_4/TensorArrayV2Stack/TensorListStack�
"simple_rnn_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2$
"simple_rnn_4/strided_slice_3/stack�
$simple_rnn_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$simple_rnn_4/strided_slice_3/stack_1�
$simple_rnn_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_4/strided_slice_3/stack_2�
simple_rnn_4/strided_slice_3StridedSlice8simple_rnn_4/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_4/strided_slice_3/stack:output:0-simple_rnn_4/strided_slice_3/stack_1:output:0-simple_rnn_4/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
simple_rnn_4/strided_slice_3�
simple_rnn_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_4/transpose_1/perm�
simple_rnn_4/transpose_1	Transpose8simple_rnn_4/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_4/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
simple_rnn_4/transpose_1t
simple_rnn_5/ShapeShapesimple_rnn_4/transpose_1:y:0*
T0*
_output_shapes
:2
simple_rnn_5/Shape�
 simple_rnn_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 simple_rnn_5/strided_slice/stack�
"simple_rnn_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_5/strided_slice/stack_1�
"simple_rnn_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_5/strided_slice/stack_2�
simple_rnn_5/strided_sliceStridedSlicesimple_rnn_5/Shape:output:0)simple_rnn_5/strided_slice/stack:output:0+simple_rnn_5/strided_slice/stack_1:output:0+simple_rnn_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_5/strided_slicev
simple_rnn_5/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn_5/zeros/mul/y�
simple_rnn_5/zeros/mulMul#simple_rnn_5/strided_slice:output:0!simple_rnn_5/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_5/zeros/muly
simple_rnn_5/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
simple_rnn_5/zeros/Less/y�
simple_rnn_5/zeros/LessLesssimple_rnn_5/zeros/mul:z:0"simple_rnn_5/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_5/zeros/Less|
simple_rnn_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn_5/zeros/packed/1�
simple_rnn_5/zeros/packedPack#simple_rnn_5/strided_slice:output:0$simple_rnn_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
simple_rnn_5/zeros/packedy
simple_rnn_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
simple_rnn_5/zeros/Const�
simple_rnn_5/zerosFill"simple_rnn_5/zeros/packed:output:0!simple_rnn_5/zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_5/zeros�
simple_rnn_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_5/transpose/perm�
simple_rnn_5/transpose	Transposesimple_rnn_4/transpose_1:y:0$simple_rnn_5/transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
simple_rnn_5/transposev
simple_rnn_5/Shape_1Shapesimple_rnn_5/transpose:y:0*
T0*
_output_shapes
:2
simple_rnn_5/Shape_1�
"simple_rnn_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn_5/strided_slice_1/stack�
$simple_rnn_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_5/strided_slice_1/stack_1�
$simple_rnn_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_5/strided_slice_1/stack_2�
simple_rnn_5/strided_slice_1StridedSlicesimple_rnn_5/Shape_1:output:0+simple_rnn_5/strided_slice_1/stack:output:0-simple_rnn_5/strided_slice_1/stack_1:output:0-simple_rnn_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_5/strided_slice_1�
(simple_rnn_5/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2*
(simple_rnn_5/TensorArrayV2/element_shape�
simple_rnn_5/TensorArrayV2TensorListReserve1simple_rnn_5/TensorArrayV2/element_shape:output:0%simple_rnn_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_5/TensorArrayV2�
Bsimple_rnn_5/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2D
Bsimple_rnn_5/TensorArrayUnstack/TensorListFromTensor/element_shape�
4simple_rnn_5/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_5/transpose:y:0Ksimple_rnn_5/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type026
4simple_rnn_5/TensorArrayUnstack/TensorListFromTensor�
"simple_rnn_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn_5/strided_slice_2/stack�
$simple_rnn_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_5/strided_slice_2/stack_1�
$simple_rnn_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_5/strided_slice_2/stack_2�
simple_rnn_5/strided_slice_2StridedSlicesimple_rnn_5/transpose:y:0+simple_rnn_5/strided_slice_2/stack:output:0-simple_rnn_5/strided_slice_2/stack_1:output:0-simple_rnn_5/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
simple_rnn_5/strided_slice_2�
4simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp=simple_rnn_5_simple_rnn_cell_5_matmul_readvariableop_resource*
_output_shapes

:  *
dtype026
4simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOp�
%simple_rnn_5/simple_rnn_cell_5/MatMulMatMul%simple_rnn_5/strided_slice_2:output:0<simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2'
%simple_rnn_5/simple_rnn_cell_5/MatMul�
5simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp>simple_rnn_5_simple_rnn_cell_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype027
5simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOp�
&simple_rnn_5/simple_rnn_cell_5/BiasAddBiasAdd/simple_rnn_5/simple_rnn_cell_5/MatMul:product:0=simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2(
&simple_rnn_5/simple_rnn_cell_5/BiasAdd�
6simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp?simple_rnn_5_simple_rnn_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype028
6simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp�
'simple_rnn_5/simple_rnn_cell_5/MatMul_1MatMulsimple_rnn_5/zeros:output:0>simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2)
'simple_rnn_5/simple_rnn_cell_5/MatMul_1�
"simple_rnn_5/simple_rnn_cell_5/addAddV2/simple_rnn_5/simple_rnn_cell_5/BiasAdd:output:01simple_rnn_5/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2$
"simple_rnn_5/simple_rnn_cell_5/add�
#simple_rnn_5/simple_rnn_cell_5/TanhTanh&simple_rnn_5/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:��������� 2%
#simple_rnn_5/simple_rnn_cell_5/Tanh�
*simple_rnn_5/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2,
*simple_rnn_5/TensorArrayV2_1/element_shape�
simple_rnn_5/TensorArrayV2_1TensorListReserve3simple_rnn_5/TensorArrayV2_1/element_shape:output:0%simple_rnn_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_5/TensorArrayV2_1h
simple_rnn_5/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn_5/time�
%simple_rnn_5/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%simple_rnn_5/while/maximum_iterations�
simple_rnn_5/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2!
simple_rnn_5/while/loop_counter�
simple_rnn_5/whileWhile(simple_rnn_5/while/loop_counter:output:0.simple_rnn_5/while/maximum_iterations:output:0simple_rnn_5/time:output:0%simple_rnn_5/TensorArrayV2_1:handle:0simple_rnn_5/zeros:output:0%simple_rnn_5/strided_slice_1:output:0Dsimple_rnn_5/TensorArrayUnstack/TensorListFromTensor:output_handle:0=simple_rnn_5_simple_rnn_cell_5_matmul_readvariableop_resource>simple_rnn_5_simple_rnn_cell_5_biasadd_readvariableop_resource?simple_rnn_5_simple_rnn_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :��������� : : : : : *%
_read_only_resource_inputs
	**
body"R 
simple_rnn_5_while_body_100748**
cond"R 
simple_rnn_5_while_cond_100747*8
output_shapes'
%: : : : :��������� : : : : : *
parallel_iterations 2
simple_rnn_5/while�
=simple_rnn_5/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2?
=simple_rnn_5/TensorArrayV2Stack/TensorListStack/element_shape�
/simple_rnn_5/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_5/while:output:3Fsimple_rnn_5/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������ *
element_dtype021
/simple_rnn_5/TensorArrayV2Stack/TensorListStack�
"simple_rnn_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2$
"simple_rnn_5/strided_slice_3/stack�
$simple_rnn_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$simple_rnn_5/strided_slice_3/stack_1�
$simple_rnn_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_5/strided_slice_3/stack_2�
simple_rnn_5/strided_slice_3StridedSlice8simple_rnn_5/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_5/strided_slice_3/stack:output:0-simple_rnn_5/strided_slice_3/stack_1:output:0-simple_rnn_5/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
simple_rnn_5/strided_slice_3�
simple_rnn_5/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_5/transpose_1/perm�
simple_rnn_5/transpose_1	Transpose8simple_rnn_5/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_5/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
simple_rnn_5/transpose_1�
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_22/MatMul/ReadVariableOp�
dense_22/MatMulMatMul%simple_rnn_5/strided_slice_3:output:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_22/MatMul�
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_22/BiasAdd/ReadVariableOp�
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_22/BiasAdds
dense_22/ReluReludense_22/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_22/Relu�
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_23/MatMul/ReadVariableOp�
dense_23/MatMulMatMuldense_22/Relu:activations:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_23/MatMul�
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_23/BiasAdd/ReadVariableOp�
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_23/BiasAdd�
IdentityIdentitydense_23/BiasAdd:output:0 ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp6^simple_rnn_3/simple_rnn_cell_3/BiasAdd/ReadVariableOp5^simple_rnn_3/simple_rnn_cell_3/MatMul/ReadVariableOp7^simple_rnn_3/simple_rnn_cell_3/MatMul_1/ReadVariableOp^simple_rnn_3/while6^simple_rnn_4/simple_rnn_cell_4/BiasAdd/ReadVariableOp5^simple_rnn_4/simple_rnn_cell_4/MatMul/ReadVariableOp7^simple_rnn_4/simple_rnn_cell_4/MatMul_1/ReadVariableOp^simple_rnn_4/while6^simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOp5^simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOp7^simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp^simple_rnn_5/while*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*c
_input_shapesR
P:������������������:::::::::::::2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2n
5simple_rnn_3/simple_rnn_cell_3/BiasAdd/ReadVariableOp5simple_rnn_3/simple_rnn_cell_3/BiasAdd/ReadVariableOp2l
4simple_rnn_3/simple_rnn_cell_3/MatMul/ReadVariableOp4simple_rnn_3/simple_rnn_cell_3/MatMul/ReadVariableOp2p
6simple_rnn_3/simple_rnn_cell_3/MatMul_1/ReadVariableOp6simple_rnn_3/simple_rnn_cell_3/MatMul_1/ReadVariableOp2(
simple_rnn_3/whilesimple_rnn_3/while2n
5simple_rnn_4/simple_rnn_cell_4/BiasAdd/ReadVariableOp5simple_rnn_4/simple_rnn_cell_4/BiasAdd/ReadVariableOp2l
4simple_rnn_4/simple_rnn_cell_4/MatMul/ReadVariableOp4simple_rnn_4/simple_rnn_cell_4/MatMul/ReadVariableOp2p
6simple_rnn_4/simple_rnn_cell_4/MatMul_1/ReadVariableOp6simple_rnn_4/simple_rnn_cell_4/MatMul_1/ReadVariableOp2(
simple_rnn_4/whilesimple_rnn_4/while2n
5simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOp5simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOp2l
4simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOp4simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOp2p
6simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp6simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp2(
simple_rnn_5/whilesimple_rnn_5/while:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�<
�
G__inference_simple_rnn_4_layer_call_and_return_conditional_losses_98449

inputs
simple_rnn_cell_4_98374
simple_rnn_cell_4_98376
simple_rnn_cell_4_98378
identity��)simple_rnn_cell_4/StatefulPartitionedCall�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
strided_slice_2�
)simple_rnn_cell_4/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_4_98374simple_rnn_cell_4_98376simple_rnn_cell_4_98378*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:��������� :��������� *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_981122+
)simple_rnn_cell_4/StatefulPartitionedCall�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_4_98374simple_rnn_cell_4_98376simple_rnn_cell_4_98378*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :��������� : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_98386*
condR
while_cond_98385*8
output_shapes'
%: : : : :��������� : : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
transpose_1�
IdentityIdentitytranspose_1:y:0*^simple_rnn_cell_4/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :������������������ 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������ :::2V
)simple_rnn_cell_4/StatefulPartitionedCall)simple_rnn_cell_4/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�

�
simple_rnn_3_while_cond_1005316
2simple_rnn_3_while_simple_rnn_3_while_loop_counter<
8simple_rnn_3_while_simple_rnn_3_while_maximum_iterations"
simple_rnn_3_while_placeholder$
 simple_rnn_3_while_placeholder_1$
 simple_rnn_3_while_placeholder_28
4simple_rnn_3_while_less_simple_rnn_3_strided_slice_1N
Jsimple_rnn_3_while_simple_rnn_3_while_cond_100531___redundant_placeholder0N
Jsimple_rnn_3_while_simple_rnn_3_while_cond_100531___redundant_placeholder1N
Jsimple_rnn_3_while_simple_rnn_3_while_cond_100531___redundant_placeholder2N
Jsimple_rnn_3_while_simple_rnn_3_while_cond_100531___redundant_placeholder3
simple_rnn_3_while_identity
�
simple_rnn_3/while/LessLesssimple_rnn_3_while_placeholder4simple_rnn_3_while_less_simple_rnn_3_strided_slice_1*
T0*
_output_shapes
: 2
simple_rnn_3/while/Less�
simple_rnn_3/while/IdentityIdentitysimple_rnn_3/while/Less:z:0*
T0
*
_output_shapes
: 2
simple_rnn_3/while/Identity"C
simple_rnn_3_while_identity$simple_rnn_3/while/Identity:output:0*@
_input_shapes/
-: : : : :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�	
�
C__inference_dense_22_layer_call_and_return_conditional_losses_99886

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�<
�
G__inference_simple_rnn_4_layer_call_and_return_conditional_losses_98566

inputs
simple_rnn_cell_4_98491
simple_rnn_cell_4_98493
simple_rnn_cell_4_98495
identity��)simple_rnn_cell_4/StatefulPartitionedCall�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
strided_slice_2�
)simple_rnn_cell_4/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_4_98491simple_rnn_cell_4_98493simple_rnn_cell_4_98495*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:��������� :��������� *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_981292+
)simple_rnn_cell_4/StatefulPartitionedCall�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_4_98491simple_rnn_cell_4_98493simple_rnn_cell_4_98495*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :��������� : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_98503*
condR
while_cond_98502*8
output_shapes'
%: : : : :��������� : : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
transpose_1�
IdentityIdentitytranspose_1:y:0*^simple_rnn_cell_4/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :������������������ 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������ :::2V
)simple_rnn_cell_4/StatefulPartitionedCall)simple_rnn_cell_4/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�
~
)__inference_dense_23_layer_call_fn_102426

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_23_layer_call_and_return_conditional_losses_999122
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�#
�
while_body_98503
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0#
while_simple_rnn_cell_4_98525_0#
while_simple_rnn_cell_4_98527_0#
while_simple_rnn_cell_4_98529_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor!
while_simple_rnn_cell_4_98525!
while_simple_rnn_cell_4_98527!
while_simple_rnn_cell_4_98529��/while/simple_rnn_cell_4/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:��������� *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
/while/simple_rnn_cell_4/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_simple_rnn_cell_4_98525_0while_simple_rnn_cell_4_98527_0while_simple_rnn_cell_4_98529_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:��������� :��������� *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_9812921
/while/simple_rnn_cell_4/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder8while/simple_rnn_cell_4/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:00^while/simple_rnn_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations0^while/simple_rnn_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:00^while/simple_rnn_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/simple_rnn_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity8while/simple_rnn_cell_4/StatefulPartitionedCall:output:10^while/simple_rnn_cell_4/StatefulPartitionedCall*
T0*'
_output_shapes
:��������� 2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"@
while_simple_rnn_cell_4_98525while_simple_rnn_cell_4_98525_0"@
while_simple_rnn_cell_4_98527while_simple_rnn_cell_4_98527_0"@
while_simple_rnn_cell_4_98529while_simple_rnn_cell_4_98529_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :��������� : : :::2b
/while/simple_rnn_cell_4/StatefulPartitionedCall/while/simple_rnn_cell_4/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�
_
C__inference_lambda_1_layer_call_and_return_conditional_losses_99097

inputs
identityk
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
ExpandDims/dim�

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*4
_output_shapes"
 :������������������2

ExpandDimst
IdentityIdentityExpandDims:output:0*
T0*4
_output_shapes"
 :������������������2

Identity"
identityIdentity:output:0*/
_input_shapes
:������������������:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�	
�
2__inference_simple_rnn_cell_4_layer_call_fn_102536

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:��������� :��������� *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_981122
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:��������� 2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:��������� 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:��������� :��������� :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states/0
�
�
-__inference_simple_rnn_3_layer_call_fn_101146
inputs_0
unknown
	unknown_0
	unknown_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_979372
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :������������������ 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�
�
while_cond_101560
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_101560___redundant_placeholder04
0while_while_cond_101560___redundant_placeholder14
0while_while_cond_101560___redundant_placeholder24
0while_while_cond_101560___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�#
�
while_body_99015
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0#
while_simple_rnn_cell_5_99037_0#
while_simple_rnn_cell_5_99039_0#
while_simple_rnn_cell_5_99041_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor!
while_simple_rnn_cell_5_99037!
while_simple_rnn_cell_5_99039!
while_simple_rnn_cell_5_99041��/while/simple_rnn_cell_5/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:��������� *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
/while/simple_rnn_cell_5/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_simple_rnn_cell_5_99037_0while_simple_rnn_cell_5_99039_0while_simple_rnn_cell_5_99041_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:��������� :��������� *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_9864121
/while/simple_rnn_cell_5/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder8while/simple_rnn_cell_5/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:00^while/simple_rnn_cell_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations0^while/simple_rnn_cell_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:00^while/simple_rnn_cell_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/simple_rnn_cell_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity8while/simple_rnn_cell_5/StatefulPartitionedCall:output:10^while/simple_rnn_cell_5/StatefulPartitionedCall*
T0*'
_output_shapes
:��������� 2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"@
while_simple_rnn_cell_5_99037while_simple_rnn_cell_5_99037_0"@
while_simple_rnn_cell_5_99039while_simple_rnn_cell_5_99039_0"@
while_simple_rnn_cell_5_99041while_simple_rnn_cell_5_99041_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :��������� : : :::2b
/while/simple_rnn_cell_5/StatefulPartitionedCall/while/simple_rnn_cell_5/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�
�
-__inference_simple_rnn_4_layer_call_fn_101884
inputs_0
unknown
	unknown_0
	unknown_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_simple_rnn_4_layer_call_and_return_conditional_losses_984492
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :������������������ 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������ :::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������ 
"
_user_specified_name
inputs/0
�	
�
-__inference_sequential_6_layer_call_fn_100889

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_sequential_6_layer_call_and_return_conditional_losses_1000712
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*c
_input_shapesR
P:������������������:::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
�
M__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_102522

inputs
states_0"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2	
BiasAdd�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2

MatMul_1k
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2
addO
TanhTanhadd:z:0*
T0*'
_output_shapes
:��������� 2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2

Identity�

Identity_1IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:��������� :��������� :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states/0
�!
�
G__inference_sequential_6_layer_call_and_return_conditional_losses_99965
lambda_1_input
simple_rnn_3_99933
simple_rnn_3_99935
simple_rnn_3_99937
simple_rnn_4_99940
simple_rnn_4_99942
simple_rnn_4_99944
simple_rnn_5_99947
simple_rnn_5_99949
simple_rnn_5_99951
dense_22_99954
dense_22_99956
dense_23_99959
dense_23_99961
identity�� dense_22/StatefulPartitionedCall� dense_23/StatefulPartitionedCall�$simple_rnn_3/StatefulPartitionedCall�$simple_rnn_4/StatefulPartitionedCall�$simple_rnn_5/StatefulPartitionedCall�
lambda_1/PartitionedCallPartitionedCalllambda_1_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_lambda_1_layer_call_and_return_conditional_losses_991032
lambda_1/PartitionedCall�
$simple_rnn_3/StatefulPartitionedCallStatefulPartitionedCall!lambda_1/PartitionedCall:output:0simple_rnn_3_99933simple_rnn_3_99935simple_rnn_3_99937*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_993392&
$simple_rnn_3/StatefulPartitionedCall�
$simple_rnn_4/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_3/StatefulPartitionedCall:output:0simple_rnn_4_99940simple_rnn_4_99942simple_rnn_4_99944*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_simple_rnn_4_layer_call_and_return_conditional_losses_995922&
$simple_rnn_4/StatefulPartitionedCall�
$simple_rnn_5/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_4/StatefulPartitionedCall:output:0simple_rnn_5_99947simple_rnn_5_99949simple_rnn_5_99951*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_simple_rnn_5_layer_call_and_return_conditional_losses_998452&
$simple_rnn_5/StatefulPartitionedCall�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_5/StatefulPartitionedCall:output:0dense_22_99954dense_22_99956*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_22_layer_call_and_return_conditional_losses_998862"
 dense_22/StatefulPartitionedCall�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_99959dense_23_99961*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_23_layer_call_and_return_conditional_losses_999122"
 dense_23/StatefulPartitionedCall�
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall%^simple_rnn_3/StatefulPartitionedCall%^simple_rnn_4/StatefulPartitionedCall%^simple_rnn_5/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*c
_input_shapesR
P:������������������:::::::::::::2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2L
$simple_rnn_3/StatefulPartitionedCall$simple_rnn_3/StatefulPartitionedCall2L
$simple_rnn_4/StatefulPartitionedCall$simple_rnn_4/StatefulPartitionedCall2L
$simple_rnn_5/StatefulPartitionedCall$simple_rnn_5/StatefulPartitionedCall:` \
0
_output_shapes
:������������������
(
_user_specified_namelambda_1_input
�3
�
while_body_99526
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_4_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_4_matmul_readvariableop_resource;
7while_simple_rnn_cell_4_biasadd_readvariableop_resource<
8while_simple_rnn_cell_4_matmul_1_readvariableop_resource��.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp�-while/simple_rnn_cell_4/MatMul/ReadVariableOp�/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:��������� *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
-while/simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_4_matmul_readvariableop_resource_0*
_output_shapes

:  *
dtype02/
-while/simple_rnn_cell_4/MatMul/ReadVariableOp�
while/simple_rnn_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2 
while/simple_rnn_cell_4/MatMul�
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype020
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp�
while/simple_rnn_cell_4/BiasAddBiasAdd(while/simple_rnn_cell_4/MatMul:product:06while/simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2!
while/simple_rnn_cell_4/BiasAdd�
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype021
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp�
 while/simple_rnn_cell_4/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2"
 while/simple_rnn_cell_4/MatMul_1�
while/simple_rnn_cell_4/addAddV2(while/simple_rnn_cell_4/BiasAdd:output:0*while/simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2
while/simple_rnn_cell_4/add�
while/simple_rnn_cell_4/TanhTanhwhile/simple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:��������� 2
while/simple_rnn_cell_4/Tanh�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_4/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity while/simple_rnn_cell_4/Tanh:y:0/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_4_biasadd_readvariableop_resource9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_4_matmul_1_readvariableop_resource:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_4_matmul_readvariableop_resource8while_simple_rnn_cell_4_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :��������� : : :::2`
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_4/MatMul/ReadVariableOp-while/simple_rnn_cell_4/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�Q
�
*sequential_6_simple_rnn_3_while_body_97256P
Lsequential_6_simple_rnn_3_while_sequential_6_simple_rnn_3_while_loop_counterV
Rsequential_6_simple_rnn_3_while_sequential_6_simple_rnn_3_while_maximum_iterations/
+sequential_6_simple_rnn_3_while_placeholder1
-sequential_6_simple_rnn_3_while_placeholder_11
-sequential_6_simple_rnn_3_while_placeholder_2O
Ksequential_6_simple_rnn_3_while_sequential_6_simple_rnn_3_strided_slice_1_0�
�sequential_6_simple_rnn_3_while_tensorarrayv2read_tensorlistgetitem_sequential_6_simple_rnn_3_tensorarrayunstack_tensorlistfromtensor_0V
Rsequential_6_simple_rnn_3_while_simple_rnn_cell_3_matmul_readvariableop_resource_0W
Ssequential_6_simple_rnn_3_while_simple_rnn_cell_3_biasadd_readvariableop_resource_0X
Tsequential_6_simple_rnn_3_while_simple_rnn_cell_3_matmul_1_readvariableop_resource_0,
(sequential_6_simple_rnn_3_while_identity.
*sequential_6_simple_rnn_3_while_identity_1.
*sequential_6_simple_rnn_3_while_identity_2.
*sequential_6_simple_rnn_3_while_identity_3.
*sequential_6_simple_rnn_3_while_identity_4M
Isequential_6_simple_rnn_3_while_sequential_6_simple_rnn_3_strided_slice_1�
�sequential_6_simple_rnn_3_while_tensorarrayv2read_tensorlistgetitem_sequential_6_simple_rnn_3_tensorarrayunstack_tensorlistfromtensorT
Psequential_6_simple_rnn_3_while_simple_rnn_cell_3_matmul_readvariableop_resourceU
Qsequential_6_simple_rnn_3_while_simple_rnn_cell_3_biasadd_readvariableop_resourceV
Rsequential_6_simple_rnn_3_while_simple_rnn_cell_3_matmul_1_readvariableop_resource��Hsequential_6/simple_rnn_3/while/simple_rnn_cell_3/BiasAdd/ReadVariableOp�Gsequential_6/simple_rnn_3/while/simple_rnn_cell_3/MatMul/ReadVariableOp�Isequential_6/simple_rnn_3/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp�
Qsequential_6/simple_rnn_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2S
Qsequential_6/simple_rnn_3/while/TensorArrayV2Read/TensorListGetItem/element_shape�
Csequential_6/simple_rnn_3/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem�sequential_6_simple_rnn_3_while_tensorarrayv2read_tensorlistgetitem_sequential_6_simple_rnn_3_tensorarrayunstack_tensorlistfromtensor_0+sequential_6_simple_rnn_3_while_placeholderZsequential_6/simple_rnn_3/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02E
Csequential_6/simple_rnn_3/while/TensorArrayV2Read/TensorListGetItem�
Gsequential_6/simple_rnn_3/while/simple_rnn_cell_3/MatMul/ReadVariableOpReadVariableOpRsequential_6_simple_rnn_3_while_simple_rnn_cell_3_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype02I
Gsequential_6/simple_rnn_3/while/simple_rnn_cell_3/MatMul/ReadVariableOp�
8sequential_6/simple_rnn_3/while/simple_rnn_cell_3/MatMulMatMulJsequential_6/simple_rnn_3/while/TensorArrayV2Read/TensorListGetItem:item:0Osequential_6/simple_rnn_3/while/simple_rnn_cell_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2:
8sequential_6/simple_rnn_3/while/simple_rnn_cell_3/MatMul�
Hsequential_6/simple_rnn_3/while/simple_rnn_cell_3/BiasAdd/ReadVariableOpReadVariableOpSsequential_6_simple_rnn_3_while_simple_rnn_cell_3_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype02J
Hsequential_6/simple_rnn_3/while/simple_rnn_cell_3/BiasAdd/ReadVariableOp�
9sequential_6/simple_rnn_3/while/simple_rnn_cell_3/BiasAddBiasAddBsequential_6/simple_rnn_3/while/simple_rnn_cell_3/MatMul:product:0Psequential_6/simple_rnn_3/while/simple_rnn_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2;
9sequential_6/simple_rnn_3/while/simple_rnn_cell_3/BiasAdd�
Isequential_6/simple_rnn_3/while/simple_rnn_cell_3/MatMul_1/ReadVariableOpReadVariableOpTsequential_6_simple_rnn_3_while_simple_rnn_cell_3_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype02K
Isequential_6/simple_rnn_3/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp�
:sequential_6/simple_rnn_3/while/simple_rnn_cell_3/MatMul_1MatMul-sequential_6_simple_rnn_3_while_placeholder_2Qsequential_6/simple_rnn_3/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2<
:sequential_6/simple_rnn_3/while/simple_rnn_cell_3/MatMul_1�
5sequential_6/simple_rnn_3/while/simple_rnn_cell_3/addAddV2Bsequential_6/simple_rnn_3/while/simple_rnn_cell_3/BiasAdd:output:0Dsequential_6/simple_rnn_3/while/simple_rnn_cell_3/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 27
5sequential_6/simple_rnn_3/while/simple_rnn_cell_3/add�
6sequential_6/simple_rnn_3/while/simple_rnn_cell_3/TanhTanh9sequential_6/simple_rnn_3/while/simple_rnn_cell_3/add:z:0*
T0*'
_output_shapes
:��������� 28
6sequential_6/simple_rnn_3/while/simple_rnn_cell_3/Tanh�
Dsequential_6/simple_rnn_3/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem-sequential_6_simple_rnn_3_while_placeholder_1+sequential_6_simple_rnn_3_while_placeholder:sequential_6/simple_rnn_3/while/simple_rnn_cell_3/Tanh:y:0*
_output_shapes
: *
element_dtype02F
Dsequential_6/simple_rnn_3/while/TensorArrayV2Write/TensorListSetItem�
%sequential_6/simple_rnn_3/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_6/simple_rnn_3/while/add/y�
#sequential_6/simple_rnn_3/while/addAddV2+sequential_6_simple_rnn_3_while_placeholder.sequential_6/simple_rnn_3/while/add/y:output:0*
T0*
_output_shapes
: 2%
#sequential_6/simple_rnn_3/while/add�
'sequential_6/simple_rnn_3/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_6/simple_rnn_3/while/add_1/y�
%sequential_6/simple_rnn_3/while/add_1AddV2Lsequential_6_simple_rnn_3_while_sequential_6_simple_rnn_3_while_loop_counter0sequential_6/simple_rnn_3/while/add_1/y:output:0*
T0*
_output_shapes
: 2'
%sequential_6/simple_rnn_3/while/add_1�
(sequential_6/simple_rnn_3/while/IdentityIdentity)sequential_6/simple_rnn_3/while/add_1:z:0I^sequential_6/simple_rnn_3/while/simple_rnn_cell_3/BiasAdd/ReadVariableOpH^sequential_6/simple_rnn_3/while/simple_rnn_cell_3/MatMul/ReadVariableOpJ^sequential_6/simple_rnn_3/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2*
(sequential_6/simple_rnn_3/while/Identity�
*sequential_6/simple_rnn_3/while/Identity_1IdentityRsequential_6_simple_rnn_3_while_sequential_6_simple_rnn_3_while_maximum_iterationsI^sequential_6/simple_rnn_3/while/simple_rnn_cell_3/BiasAdd/ReadVariableOpH^sequential_6/simple_rnn_3/while/simple_rnn_cell_3/MatMul/ReadVariableOpJ^sequential_6/simple_rnn_3/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2,
*sequential_6/simple_rnn_3/while/Identity_1�
*sequential_6/simple_rnn_3/while/Identity_2Identity'sequential_6/simple_rnn_3/while/add:z:0I^sequential_6/simple_rnn_3/while/simple_rnn_cell_3/BiasAdd/ReadVariableOpH^sequential_6/simple_rnn_3/while/simple_rnn_cell_3/MatMul/ReadVariableOpJ^sequential_6/simple_rnn_3/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2,
*sequential_6/simple_rnn_3/while/Identity_2�
*sequential_6/simple_rnn_3/while/Identity_3IdentityTsequential_6/simple_rnn_3/while/TensorArrayV2Write/TensorListSetItem:output_handle:0I^sequential_6/simple_rnn_3/while/simple_rnn_cell_3/BiasAdd/ReadVariableOpH^sequential_6/simple_rnn_3/while/simple_rnn_cell_3/MatMul/ReadVariableOpJ^sequential_6/simple_rnn_3/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2,
*sequential_6/simple_rnn_3/while/Identity_3�
*sequential_6/simple_rnn_3/while/Identity_4Identity:sequential_6/simple_rnn_3/while/simple_rnn_cell_3/Tanh:y:0I^sequential_6/simple_rnn_3/while/simple_rnn_cell_3/BiasAdd/ReadVariableOpH^sequential_6/simple_rnn_3/while/simple_rnn_cell_3/MatMul/ReadVariableOpJ^sequential_6/simple_rnn_3/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2,
*sequential_6/simple_rnn_3/while/Identity_4"]
(sequential_6_simple_rnn_3_while_identity1sequential_6/simple_rnn_3/while/Identity:output:0"a
*sequential_6_simple_rnn_3_while_identity_13sequential_6/simple_rnn_3/while/Identity_1:output:0"a
*sequential_6_simple_rnn_3_while_identity_23sequential_6/simple_rnn_3/while/Identity_2:output:0"a
*sequential_6_simple_rnn_3_while_identity_33sequential_6/simple_rnn_3/while/Identity_3:output:0"a
*sequential_6_simple_rnn_3_while_identity_43sequential_6/simple_rnn_3/while/Identity_4:output:0"�
Isequential_6_simple_rnn_3_while_sequential_6_simple_rnn_3_strided_slice_1Ksequential_6_simple_rnn_3_while_sequential_6_simple_rnn_3_strided_slice_1_0"�
Qsequential_6_simple_rnn_3_while_simple_rnn_cell_3_biasadd_readvariableop_resourceSsequential_6_simple_rnn_3_while_simple_rnn_cell_3_biasadd_readvariableop_resource_0"�
Rsequential_6_simple_rnn_3_while_simple_rnn_cell_3_matmul_1_readvariableop_resourceTsequential_6_simple_rnn_3_while_simple_rnn_cell_3_matmul_1_readvariableop_resource_0"�
Psequential_6_simple_rnn_3_while_simple_rnn_cell_3_matmul_readvariableop_resourceRsequential_6_simple_rnn_3_while_simple_rnn_cell_3_matmul_readvariableop_resource_0"�
�sequential_6_simple_rnn_3_while_tensorarrayv2read_tensorlistgetitem_sequential_6_simple_rnn_3_tensorarrayunstack_tensorlistfromtensor�sequential_6_simple_rnn_3_while_tensorarrayv2read_tensorlistgetitem_sequential_6_simple_rnn_3_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :��������� : : :::2�
Hsequential_6/simple_rnn_3/while/simple_rnn_cell_3/BiasAdd/ReadVariableOpHsequential_6/simple_rnn_3/while/simple_rnn_cell_3/BiasAdd/ReadVariableOp2�
Gsequential_6/simple_rnn_3/while/simple_rnn_cell_3/MatMul/ReadVariableOpGsequential_6/simple_rnn_3/while/simple_rnn_cell_3/MatMul/ReadVariableOp2�
Isequential_6/simple_rnn_3/while/simple_rnn_cell_3/MatMul_1/ReadVariableOpIsequential_6/simple_rnn_3/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_97990
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_97990___redundant_placeholder03
/while_while_cond_97990___redundant_placeholder13
/while_while_cond_97990___redundant_placeholder23
/while_while_cond_97990___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�	
�
2__inference_simple_rnn_cell_5_layer_call_fn_102612

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:��������� :��������� *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_986412
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:��������� 2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:��������� 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:��������� :��������� :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states/0
�

�
simple_rnn_5_while_cond_1004046
2simple_rnn_5_while_simple_rnn_5_while_loop_counter<
8simple_rnn_5_while_simple_rnn_5_while_maximum_iterations"
simple_rnn_5_while_placeholder$
 simple_rnn_5_while_placeholder_1$
 simple_rnn_5_while_placeholder_28
4simple_rnn_5_while_less_simple_rnn_5_strided_slice_1N
Jsimple_rnn_5_while_simple_rnn_5_while_cond_100404___redundant_placeholder0N
Jsimple_rnn_5_while_simple_rnn_5_while_cond_100404___redundant_placeholder1N
Jsimple_rnn_5_while_simple_rnn_5_while_cond_100404___redundant_placeholder2N
Jsimple_rnn_5_while_simple_rnn_5_while_cond_100404___redundant_placeholder3
simple_rnn_5_while_identity
�
simple_rnn_5/while/LessLesssimple_rnn_5_while_placeholder4simple_rnn_5_while_less_simple_rnn_5_strided_slice_1*
T0*
_output_shapes
: 2
simple_rnn_5/while/Less�
simple_rnn_5/while/IdentityIdentitysimple_rnn_5/while/Less:z:0*
T0
*
_output_shapes
: 2
simple_rnn_5/while/Identity"C
simple_rnn_5_while_identity$simple_rnn_5/while/Identity:output:0*@
_input_shapes/
-: : : : :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�B
�
simple_rnn_3_while_body_1001896
2simple_rnn_3_while_simple_rnn_3_while_loop_counter<
8simple_rnn_3_while_simple_rnn_3_while_maximum_iterations"
simple_rnn_3_while_placeholder$
 simple_rnn_3_while_placeholder_1$
 simple_rnn_3_while_placeholder_25
1simple_rnn_3_while_simple_rnn_3_strided_slice_1_0q
msimple_rnn_3_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_3_tensorarrayunstack_tensorlistfromtensor_0I
Esimple_rnn_3_while_simple_rnn_cell_3_matmul_readvariableop_resource_0J
Fsimple_rnn_3_while_simple_rnn_cell_3_biasadd_readvariableop_resource_0K
Gsimple_rnn_3_while_simple_rnn_cell_3_matmul_1_readvariableop_resource_0
simple_rnn_3_while_identity!
simple_rnn_3_while_identity_1!
simple_rnn_3_while_identity_2!
simple_rnn_3_while_identity_3!
simple_rnn_3_while_identity_43
/simple_rnn_3_while_simple_rnn_3_strided_slice_1o
ksimple_rnn_3_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_3_tensorarrayunstack_tensorlistfromtensorG
Csimple_rnn_3_while_simple_rnn_cell_3_matmul_readvariableop_resourceH
Dsimple_rnn_3_while_simple_rnn_cell_3_biasadd_readvariableop_resourceI
Esimple_rnn_3_while_simple_rnn_cell_3_matmul_1_readvariableop_resource��;simple_rnn_3/while/simple_rnn_cell_3/BiasAdd/ReadVariableOp�:simple_rnn_3/while/simple_rnn_cell_3/MatMul/ReadVariableOp�<simple_rnn_3/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp�
Dsimple_rnn_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2F
Dsimple_rnn_3/while/TensorArrayV2Read/TensorListGetItem/element_shape�
6simple_rnn_3/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_3_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_3_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_3_while_placeholderMsimple_rnn_3/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype028
6simple_rnn_3/while/TensorArrayV2Read/TensorListGetItem�
:simple_rnn_3/while/simple_rnn_cell_3/MatMul/ReadVariableOpReadVariableOpEsimple_rnn_3_while_simple_rnn_cell_3_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype02<
:simple_rnn_3/while/simple_rnn_cell_3/MatMul/ReadVariableOp�
+simple_rnn_3/while/simple_rnn_cell_3/MatMulMatMul=simple_rnn_3/while/TensorArrayV2Read/TensorListGetItem:item:0Bsimple_rnn_3/while/simple_rnn_cell_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2-
+simple_rnn_3/while/simple_rnn_cell_3/MatMul�
;simple_rnn_3/while/simple_rnn_cell_3/BiasAdd/ReadVariableOpReadVariableOpFsimple_rnn_3_while_simple_rnn_cell_3_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype02=
;simple_rnn_3/while/simple_rnn_cell_3/BiasAdd/ReadVariableOp�
,simple_rnn_3/while/simple_rnn_cell_3/BiasAddBiasAdd5simple_rnn_3/while/simple_rnn_cell_3/MatMul:product:0Csimple_rnn_3/while/simple_rnn_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2.
,simple_rnn_3/while/simple_rnn_cell_3/BiasAdd�
<simple_rnn_3/while/simple_rnn_cell_3/MatMul_1/ReadVariableOpReadVariableOpGsimple_rnn_3_while_simple_rnn_cell_3_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype02>
<simple_rnn_3/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp�
-simple_rnn_3/while/simple_rnn_cell_3/MatMul_1MatMul simple_rnn_3_while_placeholder_2Dsimple_rnn_3/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2/
-simple_rnn_3/while/simple_rnn_cell_3/MatMul_1�
(simple_rnn_3/while/simple_rnn_cell_3/addAddV25simple_rnn_3/while/simple_rnn_cell_3/BiasAdd:output:07simple_rnn_3/while/simple_rnn_cell_3/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2*
(simple_rnn_3/while/simple_rnn_cell_3/add�
)simple_rnn_3/while/simple_rnn_cell_3/TanhTanh,simple_rnn_3/while/simple_rnn_cell_3/add:z:0*
T0*'
_output_shapes
:��������� 2+
)simple_rnn_3/while/simple_rnn_cell_3/Tanh�
7simple_rnn_3/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_3_while_placeholder_1simple_rnn_3_while_placeholder-simple_rnn_3/while/simple_rnn_cell_3/Tanh:y:0*
_output_shapes
: *
element_dtype029
7simple_rnn_3/while/TensorArrayV2Write/TensorListSetItemv
simple_rnn_3/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_3/while/add/y�
simple_rnn_3/while/addAddV2simple_rnn_3_while_placeholder!simple_rnn_3/while/add/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_3/while/addz
simple_rnn_3/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_3/while/add_1/y�
simple_rnn_3/while/add_1AddV22simple_rnn_3_while_simple_rnn_3_while_loop_counter#simple_rnn_3/while/add_1/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_3/while/add_1�
simple_rnn_3/while/IdentityIdentitysimple_rnn_3/while/add_1:z:0<^simple_rnn_3/while/simple_rnn_cell_3/BiasAdd/ReadVariableOp;^simple_rnn_3/while/simple_rnn_cell_3/MatMul/ReadVariableOp=^simple_rnn_3/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_3/while/Identity�
simple_rnn_3/while/Identity_1Identity8simple_rnn_3_while_simple_rnn_3_while_maximum_iterations<^simple_rnn_3/while/simple_rnn_cell_3/BiasAdd/ReadVariableOp;^simple_rnn_3/while/simple_rnn_cell_3/MatMul/ReadVariableOp=^simple_rnn_3/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_3/while/Identity_1�
simple_rnn_3/while/Identity_2Identitysimple_rnn_3/while/add:z:0<^simple_rnn_3/while/simple_rnn_cell_3/BiasAdd/ReadVariableOp;^simple_rnn_3/while/simple_rnn_cell_3/MatMul/ReadVariableOp=^simple_rnn_3/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_3/while/Identity_2�
simple_rnn_3/while/Identity_3IdentityGsimple_rnn_3/while/TensorArrayV2Write/TensorListSetItem:output_handle:0<^simple_rnn_3/while/simple_rnn_cell_3/BiasAdd/ReadVariableOp;^simple_rnn_3/while/simple_rnn_cell_3/MatMul/ReadVariableOp=^simple_rnn_3/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_3/while/Identity_3�
simple_rnn_3/while/Identity_4Identity-simple_rnn_3/while/simple_rnn_cell_3/Tanh:y:0<^simple_rnn_3/while/simple_rnn_cell_3/BiasAdd/ReadVariableOp;^simple_rnn_3/while/simple_rnn_cell_3/MatMul/ReadVariableOp=^simple_rnn_3/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2
simple_rnn_3/while/Identity_4"C
simple_rnn_3_while_identity$simple_rnn_3/while/Identity:output:0"G
simple_rnn_3_while_identity_1&simple_rnn_3/while/Identity_1:output:0"G
simple_rnn_3_while_identity_2&simple_rnn_3/while/Identity_2:output:0"G
simple_rnn_3_while_identity_3&simple_rnn_3/while/Identity_3:output:0"G
simple_rnn_3_while_identity_4&simple_rnn_3/while/Identity_4:output:0"d
/simple_rnn_3_while_simple_rnn_3_strided_slice_11simple_rnn_3_while_simple_rnn_3_strided_slice_1_0"�
Dsimple_rnn_3_while_simple_rnn_cell_3_biasadd_readvariableop_resourceFsimple_rnn_3_while_simple_rnn_cell_3_biasadd_readvariableop_resource_0"�
Esimple_rnn_3_while_simple_rnn_cell_3_matmul_1_readvariableop_resourceGsimple_rnn_3_while_simple_rnn_cell_3_matmul_1_readvariableop_resource_0"�
Csimple_rnn_3_while_simple_rnn_cell_3_matmul_readvariableop_resourceEsimple_rnn_3_while_simple_rnn_cell_3_matmul_readvariableop_resource_0"�
ksimple_rnn_3_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_3_tensorarrayunstack_tensorlistfromtensormsimple_rnn_3_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_3_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :��������� : : :::2z
;simple_rnn_3/while/simple_rnn_cell_3/BiasAdd/ReadVariableOp;simple_rnn_3/while/simple_rnn_cell_3/BiasAdd/ReadVariableOp2x
:simple_rnn_3/while/simple_rnn_cell_3/MatMul/ReadVariableOp:simple_rnn_3/while/simple_rnn_cell_3/MatMul/ReadVariableOp2|
<simple_rnn_3/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp<simple_rnn_3/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_97873
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_97873___redundant_placeholder03
/while_while_cond_97873___redundant_placeholder13
/while_while_cond_97873___redundant_placeholder23
/while_while_cond_97873___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�B
�
simple_rnn_4_while_body_1006406
2simple_rnn_4_while_simple_rnn_4_while_loop_counter<
8simple_rnn_4_while_simple_rnn_4_while_maximum_iterations"
simple_rnn_4_while_placeholder$
 simple_rnn_4_while_placeholder_1$
 simple_rnn_4_while_placeholder_25
1simple_rnn_4_while_simple_rnn_4_strided_slice_1_0q
msimple_rnn_4_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_4_tensorarrayunstack_tensorlistfromtensor_0I
Esimple_rnn_4_while_simple_rnn_cell_4_matmul_readvariableop_resource_0J
Fsimple_rnn_4_while_simple_rnn_cell_4_biasadd_readvariableop_resource_0K
Gsimple_rnn_4_while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0
simple_rnn_4_while_identity!
simple_rnn_4_while_identity_1!
simple_rnn_4_while_identity_2!
simple_rnn_4_while_identity_3!
simple_rnn_4_while_identity_43
/simple_rnn_4_while_simple_rnn_4_strided_slice_1o
ksimple_rnn_4_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_4_tensorarrayunstack_tensorlistfromtensorG
Csimple_rnn_4_while_simple_rnn_cell_4_matmul_readvariableop_resourceH
Dsimple_rnn_4_while_simple_rnn_cell_4_biasadd_readvariableop_resourceI
Esimple_rnn_4_while_simple_rnn_cell_4_matmul_1_readvariableop_resource��;simple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp�:simple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOp�<simple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp�
Dsimple_rnn_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2F
Dsimple_rnn_4/while/TensorArrayV2Read/TensorListGetItem/element_shape�
6simple_rnn_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_4_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_4_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_4_while_placeholderMsimple_rnn_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:��������� *
element_dtype028
6simple_rnn_4/while/TensorArrayV2Read/TensorListGetItem�
:simple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOpEsimple_rnn_4_while_simple_rnn_cell_4_matmul_readvariableop_resource_0*
_output_shapes

:  *
dtype02<
:simple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOp�
+simple_rnn_4/while/simple_rnn_cell_4/MatMulMatMul=simple_rnn_4/while/TensorArrayV2Read/TensorListGetItem:item:0Bsimple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2-
+simple_rnn_4/while/simple_rnn_cell_4/MatMul�
;simple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOpFsimple_rnn_4_while_simple_rnn_cell_4_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype02=
;simple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp�
,simple_rnn_4/while/simple_rnn_cell_4/BiasAddBiasAdd5simple_rnn_4/while/simple_rnn_cell_4/MatMul:product:0Csimple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2.
,simple_rnn_4/while/simple_rnn_cell_4/BiasAdd�
<simple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOpGsimple_rnn_4_while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype02>
<simple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp�
-simple_rnn_4/while/simple_rnn_cell_4/MatMul_1MatMul simple_rnn_4_while_placeholder_2Dsimple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2/
-simple_rnn_4/while/simple_rnn_cell_4/MatMul_1�
(simple_rnn_4/while/simple_rnn_cell_4/addAddV25simple_rnn_4/while/simple_rnn_cell_4/BiasAdd:output:07simple_rnn_4/while/simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2*
(simple_rnn_4/while/simple_rnn_cell_4/add�
)simple_rnn_4/while/simple_rnn_cell_4/TanhTanh,simple_rnn_4/while/simple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:��������� 2+
)simple_rnn_4/while/simple_rnn_cell_4/Tanh�
7simple_rnn_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_4_while_placeholder_1simple_rnn_4_while_placeholder-simple_rnn_4/while/simple_rnn_cell_4/Tanh:y:0*
_output_shapes
: *
element_dtype029
7simple_rnn_4/while/TensorArrayV2Write/TensorListSetItemv
simple_rnn_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_4/while/add/y�
simple_rnn_4/while/addAddV2simple_rnn_4_while_placeholder!simple_rnn_4/while/add/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_4/while/addz
simple_rnn_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_4/while/add_1/y�
simple_rnn_4/while/add_1AddV22simple_rnn_4_while_simple_rnn_4_while_loop_counter#simple_rnn_4/while/add_1/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_4/while/add_1�
simple_rnn_4/while/IdentityIdentitysimple_rnn_4/while/add_1:z:0<^simple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp;^simple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOp=^simple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_4/while/Identity�
simple_rnn_4/while/Identity_1Identity8simple_rnn_4_while_simple_rnn_4_while_maximum_iterations<^simple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp;^simple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOp=^simple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_4/while/Identity_1�
simple_rnn_4/while/Identity_2Identitysimple_rnn_4/while/add:z:0<^simple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp;^simple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOp=^simple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_4/while/Identity_2�
simple_rnn_4/while/Identity_3IdentityGsimple_rnn_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0<^simple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp;^simple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOp=^simple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_4/while/Identity_3�
simple_rnn_4/while/Identity_4Identity-simple_rnn_4/while/simple_rnn_cell_4/Tanh:y:0<^simple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp;^simple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOp=^simple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2
simple_rnn_4/while/Identity_4"C
simple_rnn_4_while_identity$simple_rnn_4/while/Identity:output:0"G
simple_rnn_4_while_identity_1&simple_rnn_4/while/Identity_1:output:0"G
simple_rnn_4_while_identity_2&simple_rnn_4/while/Identity_2:output:0"G
simple_rnn_4_while_identity_3&simple_rnn_4/while/Identity_3:output:0"G
simple_rnn_4_while_identity_4&simple_rnn_4/while/Identity_4:output:0"d
/simple_rnn_4_while_simple_rnn_4_strided_slice_11simple_rnn_4_while_simple_rnn_4_strided_slice_1_0"�
Dsimple_rnn_4_while_simple_rnn_cell_4_biasadd_readvariableop_resourceFsimple_rnn_4_while_simple_rnn_cell_4_biasadd_readvariableop_resource_0"�
Esimple_rnn_4_while_simple_rnn_cell_4_matmul_1_readvariableop_resourceGsimple_rnn_4_while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0"�
Csimple_rnn_4_while_simple_rnn_cell_4_matmul_readvariableop_resourceEsimple_rnn_4_while_simple_rnn_cell_4_matmul_readvariableop_resource_0"�
ksimple_rnn_4_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_4_tensorarrayunstack_tensorlistfromtensormsimple_rnn_4_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_4_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :��������� : : :::2z
;simple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp;simple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp2x
:simple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOp:simple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOp2|
<simple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp<simple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�	
�
C__inference_dense_23_layer_call_and_return_conditional_losses_99912

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�3
�
while_body_99667
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_5_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_5_matmul_readvariableop_resource;
7while_simple_rnn_cell_5_biasadd_readvariableop_resource<
8while_simple_rnn_cell_5_matmul_1_readvariableop_resource��.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp�-while/simple_rnn_cell_5/MatMul/ReadVariableOp�/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:��������� *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
-while/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_5_matmul_readvariableop_resource_0*
_output_shapes

:  *
dtype02/
-while/simple_rnn_cell_5/MatMul/ReadVariableOp�
while/simple_rnn_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2 
while/simple_rnn_cell_5/MatMul�
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype020
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp�
while/simple_rnn_cell_5/BiasAddBiasAdd(while/simple_rnn_cell_5/MatMul:product:06while/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2!
while/simple_rnn_cell_5/BiasAdd�
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype021
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp�
 while/simple_rnn_cell_5/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2"
 while/simple_rnn_cell_5/MatMul_1�
while/simple_rnn_cell_5/addAddV2(while/simple_rnn_cell_5/BiasAdd:output:0*while/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2
while/simple_rnn_cell_5/add�
while/simple_rnn_cell_5/TanhTanhwhile/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:��������� 2
while/simple_rnn_cell_5/Tanh�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_5/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity while/simple_rnn_cell_5/Tanh:y:0/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_5_biasadd_readvariableop_resource9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_5_matmul_1_readvariableop_resource:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_5_matmul_readvariableop_resource8while_simple_rnn_cell_5_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :��������� : : :::2`
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_5/MatMul/ReadVariableOp-while/simple_rnn_cell_5/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�	
�
-__inference_sequential_6_layer_call_fn_100100
lambda_1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllambda_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_sequential_6_layer_call_and_return_conditional_losses_1000712
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*c
_input_shapesR
P:������������������:::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
0
_output_shapes
:������������������
(
_user_specified_namelambda_1_input
�
E
)__inference_lambda_1_layer_call_fn_100911

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_lambda_1_layer_call_and_return_conditional_losses_991032
PartitionedCally
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :������������������2

Identity"
identityIdentity:output:0*/
_input_shapes
:������������������:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�H
�
G__inference_simple_rnn_5_layer_call_and_return_conditional_losses_99733

inputs4
0simple_rnn_cell_5_matmul_readvariableop_resource5
1simple_rnn_cell_5_biasadd_readvariableop_resource6
2simple_rnn_cell_5_matmul_1_readvariableop_resource
identity��(simple_rnn_cell_5/BiasAdd/ReadVariableOp�'simple_rnn_cell_5/MatMul/ReadVariableOp�)simple_rnn_cell_5/MatMul_1/ReadVariableOp�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
strided_slice_2�
'simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_5_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02)
'simple_rnn_cell_5/MatMul/ReadVariableOp�
simple_rnn_cell_5/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_5/MatMul�
(simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(simple_rnn_cell_5/BiasAdd/ReadVariableOp�
simple_rnn_cell_5/BiasAddBiasAdd"simple_rnn_cell_5/MatMul:product:00simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_5/BiasAdd�
)simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype02+
)simple_rnn_cell_5/MatMul_1/ReadVariableOp�
simple_rnn_cell_5/MatMul_1MatMulzeros:output:01simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_5/MatMul_1�
simple_rnn_cell_5/addAddV2"simple_rnn_cell_5/BiasAdd:output:0$simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_5/add�
simple_rnn_cell_5/TanhTanhsimple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_5/Tanh�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_5_matmul_readvariableop_resource1simple_rnn_cell_5_biasadd_readvariableop_resource2simple_rnn_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :��������� : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_99667*
condR
while_cond_99666*8
output_shapes'
%: : : : :��������� : : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
transpose_1�
IdentityIdentitystrided_slice_3:output:0)^simple_rnn_cell_5/BiasAdd/ReadVariableOp(^simple_rnn_cell_5/MatMul/ReadVariableOp*^simple_rnn_cell_5/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������ :::2T
(simple_rnn_cell_5/BiasAdd/ReadVariableOp(simple_rnn_cell_5/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_5/MatMul/ReadVariableOp'simple_rnn_cell_5/MatMul/ReadVariableOp2V
)simple_rnn_cell_5/MatMul_1/ReadVariableOp)simple_rnn_cell_5/MatMul_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�	
�
-__inference_sequential_6_layer_call_fn_100858

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_sequential_6_layer_call_and_return_conditional_losses_1000042
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*c
_input_shapesR
P:������������������:::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�

�
simple_rnn_4_while_cond_1002966
2simple_rnn_4_while_simple_rnn_4_while_loop_counter<
8simple_rnn_4_while_simple_rnn_4_while_maximum_iterations"
simple_rnn_4_while_placeholder$
 simple_rnn_4_while_placeholder_1$
 simple_rnn_4_while_placeholder_28
4simple_rnn_4_while_less_simple_rnn_4_strided_slice_1N
Jsimple_rnn_4_while_simple_rnn_4_while_cond_100296___redundant_placeholder0N
Jsimple_rnn_4_while_simple_rnn_4_while_cond_100296___redundant_placeholder1N
Jsimple_rnn_4_while_simple_rnn_4_while_cond_100296___redundant_placeholder2N
Jsimple_rnn_4_while_simple_rnn_4_while_cond_100296___redundant_placeholder3
simple_rnn_4_while_identity
�
simple_rnn_4/while/LessLesssimple_rnn_4_while_placeholder4simple_rnn_4_while_less_simple_rnn_4_strided_slice_1*
T0*
_output_shapes
: 2
simple_rnn_4/while/Less�
simple_rnn_4/while/IdentityIdentitysimple_rnn_4/while/Less:z:0*
T0
*
_output_shapes
: 2
simple_rnn_4/while/Identity"C
simple_rnn_4_while_identity$simple_rnn_4/while/Identity:output:0*@
_input_shapes/
-: : : : :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�
�
while_cond_100956
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_100956___redundant_placeholder04
0while_while_cond_100956___redundant_placeholder14
0while_while_cond_100956___redundant_placeholder24
0while_while_cond_100956___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�
�
while_cond_101694
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_101694___redundant_placeholder04
0while_while_cond_101694___redundant_placeholder14
0while_while_cond_101694___redundant_placeholder24
0while_while_cond_101694___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�
�
while_cond_101806
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_101806___redundant_placeholder04
0while_while_cond_101806___redundant_placeholder14
0while_while_cond_101806___redundant_placeholder24
0while_while_cond_101806___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�	
�
D__inference_dense_23_layer_call_and_return_conditional_losses_102417

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
*sequential_6_simple_rnn_4_while_cond_97363P
Lsequential_6_simple_rnn_4_while_sequential_6_simple_rnn_4_while_loop_counterV
Rsequential_6_simple_rnn_4_while_sequential_6_simple_rnn_4_while_maximum_iterations/
+sequential_6_simple_rnn_4_while_placeholder1
-sequential_6_simple_rnn_4_while_placeholder_11
-sequential_6_simple_rnn_4_while_placeholder_2R
Nsequential_6_simple_rnn_4_while_less_sequential_6_simple_rnn_4_strided_slice_1g
csequential_6_simple_rnn_4_while_sequential_6_simple_rnn_4_while_cond_97363___redundant_placeholder0g
csequential_6_simple_rnn_4_while_sequential_6_simple_rnn_4_while_cond_97363___redundant_placeholder1g
csequential_6_simple_rnn_4_while_sequential_6_simple_rnn_4_while_cond_97363___redundant_placeholder2g
csequential_6_simple_rnn_4_while_sequential_6_simple_rnn_4_while_cond_97363___redundant_placeholder3,
(sequential_6_simple_rnn_4_while_identity
�
$sequential_6/simple_rnn_4/while/LessLess+sequential_6_simple_rnn_4_while_placeholderNsequential_6_simple_rnn_4_while_less_sequential_6_simple_rnn_4_strided_slice_1*
T0*
_output_shapes
: 2&
$sequential_6/simple_rnn_4/while/Less�
(sequential_6/simple_rnn_4/while/IdentityIdentity(sequential_6/simple_rnn_4/while/Less:z:0*
T0
*
_output_shapes
: 2*
(sequential_6/simple_rnn_4/while/Identity"]
(sequential_6_simple_rnn_4_while_identity1sequential_6/simple_rnn_4/while/Identity:output:0*@
_input_shapes/
-: : : : :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�
�
-__inference_simple_rnn_3_layer_call_fn_101157
inputs_0
unknown
	unknown_0
	unknown_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_980542
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :������������������ 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�#
�
while_body_98898
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0#
while_simple_rnn_cell_5_98920_0#
while_simple_rnn_cell_5_98922_0#
while_simple_rnn_cell_5_98924_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor!
while_simple_rnn_cell_5_98920!
while_simple_rnn_cell_5_98922!
while_simple_rnn_cell_5_98924��/while/simple_rnn_cell_5/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:��������� *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
/while/simple_rnn_cell_5/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_simple_rnn_cell_5_98920_0while_simple_rnn_cell_5_98922_0while_simple_rnn_cell_5_98924_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:��������� :��������� *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_9862421
/while/simple_rnn_cell_5/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder8while/simple_rnn_cell_5/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:00^while/simple_rnn_cell_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations0^while/simple_rnn_cell_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:00^while/simple_rnn_cell_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/simple_rnn_cell_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity8while/simple_rnn_cell_5/StatefulPartitionedCall:output:10^while/simple_rnn_cell_5/StatefulPartitionedCall*
T0*'
_output_shapes
:��������� 2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"@
while_simple_rnn_cell_5_98920while_simple_rnn_cell_5_98920_0"@
while_simple_rnn_cell_5_98922while_simple_rnn_cell_5_98922_0"@
while_simple_rnn_cell_5_98924while_simple_rnn_cell_5_98924_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :��������� : : :::2b
/while/simple_rnn_cell_5/StatefulPartitionedCall/while/simple_rnn_cell_5/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�

�
simple_rnn_3_while_cond_1001886
2simple_rnn_3_while_simple_rnn_3_while_loop_counter<
8simple_rnn_3_while_simple_rnn_3_while_maximum_iterations"
simple_rnn_3_while_placeholder$
 simple_rnn_3_while_placeholder_1$
 simple_rnn_3_while_placeholder_28
4simple_rnn_3_while_less_simple_rnn_3_strided_slice_1N
Jsimple_rnn_3_while_simple_rnn_3_while_cond_100188___redundant_placeholder0N
Jsimple_rnn_3_while_simple_rnn_3_while_cond_100188___redundant_placeholder1N
Jsimple_rnn_3_while_simple_rnn_3_while_cond_100188___redundant_placeholder2N
Jsimple_rnn_3_while_simple_rnn_3_while_cond_100188___redundant_placeholder3
simple_rnn_3_while_identity
�
simple_rnn_3/while/LessLesssimple_rnn_3_while_placeholder4simple_rnn_3_while_less_simple_rnn_3_strided_slice_1*
T0*
_output_shapes
: 2
simple_rnn_3/while/Less�
simple_rnn_3/while/IdentityIdentitysimple_rnn_3/while/Less:z:0*
T0
*
_output_shapes
: 2
simple_rnn_3/while/Identity"C
simple_rnn_3_while_identity$simple_rnn_3/while/Identity:output:0*@
_input_shapes/
-: : : : :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�
�
while_cond_102298
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_102298___redundant_placeholder04
0while_while_cond_102298___redundant_placeholder14
0while_while_cond_102298___redundant_placeholder24
0while_while_cond_102298___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�
�
while_cond_101068
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_101068___redundant_placeholder04
0while_while_cond_101068___redundant_placeholder14
0while_while_cond_101068___redundant_placeholder24
0while_while_cond_101068___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�3
�
while_body_99779
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_5_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_5_matmul_readvariableop_resource;
7while_simple_rnn_cell_5_biasadd_readvariableop_resource<
8while_simple_rnn_cell_5_matmul_1_readvariableop_resource��.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp�-while/simple_rnn_cell_5/MatMul/ReadVariableOp�/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:��������� *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
-while/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_5_matmul_readvariableop_resource_0*
_output_shapes

:  *
dtype02/
-while/simple_rnn_cell_5/MatMul/ReadVariableOp�
while/simple_rnn_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2 
while/simple_rnn_cell_5/MatMul�
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype020
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp�
while/simple_rnn_cell_5/BiasAddBiasAdd(while/simple_rnn_cell_5/MatMul:product:06while/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2!
while/simple_rnn_cell_5/BiasAdd�
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype021
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp�
 while/simple_rnn_cell_5/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2"
 while/simple_rnn_cell_5/MatMul_1�
while/simple_rnn_cell_5/addAddV2(while/simple_rnn_cell_5/BiasAdd:output:0*while/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2
while/simple_rnn_cell_5/add�
while/simple_rnn_cell_5/TanhTanhwhile/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:��������� 2
while/simple_rnn_cell_5/Tanh�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_5/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity while/simple_rnn_cell_5/Tanh:y:0/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_5_biasadd_readvariableop_resource9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_5_matmul_1_readvariableop_resource:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_5_matmul_readvariableop_resource8while_simple_rnn_cell_5_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :��������� : : :::2`
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_5/MatMul/ReadVariableOp-while/simple_rnn_cell_5/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
��
�
"__inference__traced_restore_102921
file_prefix$
 assignvariableop_dense_22_kernel$
 assignvariableop_1_dense_22_bias&
"assignvariableop_2_dense_23_kernel$
 assignvariableop_3_dense_23_bias 
assignvariableop_4_adam_iter"
assignvariableop_5_adam_beta_1"
assignvariableop_6_adam_beta_2!
assignvariableop_7_adam_decay)
%assignvariableop_8_adam_learning_rate<
8assignvariableop_9_simple_rnn_3_simple_rnn_cell_3_kernelG
Cassignvariableop_10_simple_rnn_3_simple_rnn_cell_3_recurrent_kernel;
7assignvariableop_11_simple_rnn_3_simple_rnn_cell_3_bias=
9assignvariableop_12_simple_rnn_4_simple_rnn_cell_4_kernelG
Cassignvariableop_13_simple_rnn_4_simple_rnn_cell_4_recurrent_kernel;
7assignvariableop_14_simple_rnn_4_simple_rnn_cell_4_bias=
9assignvariableop_15_simple_rnn_5_simple_rnn_cell_5_kernelG
Cassignvariableop_16_simple_rnn_5_simple_rnn_cell_5_recurrent_kernel;
7assignvariableop_17_simple_rnn_5_simple_rnn_cell_5_bias
assignvariableop_18_total
assignvariableop_19_count.
*assignvariableop_20_adam_dense_22_kernel_m,
(assignvariableop_21_adam_dense_22_bias_m.
*assignvariableop_22_adam_dense_23_kernel_m,
(assignvariableop_23_adam_dense_23_bias_mD
@assignvariableop_24_adam_simple_rnn_3_simple_rnn_cell_3_kernel_mN
Jassignvariableop_25_adam_simple_rnn_3_simple_rnn_cell_3_recurrent_kernel_mB
>assignvariableop_26_adam_simple_rnn_3_simple_rnn_cell_3_bias_mD
@assignvariableop_27_adam_simple_rnn_4_simple_rnn_cell_4_kernel_mN
Jassignvariableop_28_adam_simple_rnn_4_simple_rnn_cell_4_recurrent_kernel_mB
>assignvariableop_29_adam_simple_rnn_4_simple_rnn_cell_4_bias_mD
@assignvariableop_30_adam_simple_rnn_5_simple_rnn_cell_5_kernel_mN
Jassignvariableop_31_adam_simple_rnn_5_simple_rnn_cell_5_recurrent_kernel_mB
>assignvariableop_32_adam_simple_rnn_5_simple_rnn_cell_5_bias_m.
*assignvariableop_33_adam_dense_22_kernel_v,
(assignvariableop_34_adam_dense_22_bias_v.
*assignvariableop_35_adam_dense_23_kernel_v,
(assignvariableop_36_adam_dense_23_bias_vD
@assignvariableop_37_adam_simple_rnn_3_simple_rnn_cell_3_kernel_vN
Jassignvariableop_38_adam_simple_rnn_3_simple_rnn_cell_3_recurrent_kernel_vB
>assignvariableop_39_adam_simple_rnn_3_simple_rnn_cell_3_bias_vD
@assignvariableop_40_adam_simple_rnn_4_simple_rnn_cell_4_kernel_vN
Jassignvariableop_41_adam_simple_rnn_4_simple_rnn_cell_4_recurrent_kernel_vB
>assignvariableop_42_adam_simple_rnn_4_simple_rnn_cell_4_bias_vD
@assignvariableop_43_adam_simple_rnn_5_simple_rnn_cell_5_kernel_vN
Jassignvariableop_44_adam_simple_rnn_5_simple_rnn_cell_5_recurrent_kernel_vB
>assignvariableop_45_adam_simple_rnn_5_simple_rnn_cell_5_bias_v
identity_47��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*�
value�B�/B6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*q
valuehBf/B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::*=
dtypes3
12/	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp assignvariableop_dense_22_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_22_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_23_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_23_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp8assignvariableop_9_simple_rnn_3_simple_rnn_cell_3_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOpCassignvariableop_10_simple_rnn_3_simple_rnn_cell_3_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp7assignvariableop_11_simple_rnn_3_simple_rnn_cell_3_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp9assignvariableop_12_simple_rnn_4_simple_rnn_cell_4_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOpCassignvariableop_13_simple_rnn_4_simple_rnn_cell_4_recurrent_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp7assignvariableop_14_simple_rnn_4_simple_rnn_cell_4_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp9assignvariableop_15_simple_rnn_5_simple_rnn_cell_5_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOpCassignvariableop_16_simple_rnn_5_simple_rnn_cell_5_recurrent_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp7assignvariableop_17_simple_rnn_5_simple_rnn_cell_5_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOpassignvariableop_18_totalIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOpassignvariableop_19_countIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_dense_22_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_dense_22_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp*assignvariableop_22_adam_dense_23_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_dense_23_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp@assignvariableop_24_adam_simple_rnn_3_simple_rnn_cell_3_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOpJassignvariableop_25_adam_simple_rnn_3_simple_rnn_cell_3_recurrent_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp>assignvariableop_26_adam_simple_rnn_3_simple_rnn_cell_3_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp@assignvariableop_27_adam_simple_rnn_4_simple_rnn_cell_4_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOpJassignvariableop_28_adam_simple_rnn_4_simple_rnn_cell_4_recurrent_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp>assignvariableop_29_adam_simple_rnn_4_simple_rnn_cell_4_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp@assignvariableop_30_adam_simple_rnn_5_simple_rnn_cell_5_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOpJassignvariableop_31_adam_simple_rnn_5_simple_rnn_cell_5_recurrent_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp>assignvariableop_32_adam_simple_rnn_5_simple_rnn_cell_5_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_dense_22_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_dense_22_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_23_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_23_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp@assignvariableop_37_adam_simple_rnn_3_simple_rnn_cell_3_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOpJassignvariableop_38_adam_simple_rnn_3_simple_rnn_cell_3_recurrent_kernel_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp>assignvariableop_39_adam_simple_rnn_3_simple_rnn_cell_3_bias_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp@assignvariableop_40_adam_simple_rnn_4_simple_rnn_cell_4_kernel_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOpJassignvariableop_41_adam_simple_rnn_4_simple_rnn_cell_4_recurrent_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42�
AssignVariableOp_42AssignVariableOp>assignvariableop_42_adam_simple_rnn_4_simple_rnn_cell_4_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43�
AssignVariableOp_43AssignVariableOp@assignvariableop_43_adam_simple_rnn_5_simple_rnn_cell_5_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44�
AssignVariableOp_44AssignVariableOpJassignvariableop_44_adam_simple_rnn_5_simple_rnn_cell_5_recurrent_kernel_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45�
AssignVariableOp_45AssignVariableOp>assignvariableop_45_adam_simple_rnn_5_simple_rnn_cell_5_bias_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_459
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_46Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_46�
Identity_47IdentityIdentity_46:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_47"#
identity_47Identity_47:output:0*�
_input_shapes�
�: ::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�3
�
while_body_99273
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_3_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_3_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_3_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_3_matmul_readvariableop_resource;
7while_simple_rnn_cell_3_biasadd_readvariableop_resource<
8while_simple_rnn_cell_3_matmul_1_readvariableop_resource��.while/simple_rnn_cell_3/BiasAdd/ReadVariableOp�-while/simple_rnn_cell_3/MatMul/ReadVariableOp�/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
-while/simple_rnn_cell_3/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_3_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype02/
-while/simple_rnn_cell_3/MatMul/ReadVariableOp�
while/simple_rnn_cell_3/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2 
while/simple_rnn_cell_3/MatMul�
.while/simple_rnn_cell_3/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_3_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype020
.while/simple_rnn_cell_3/BiasAdd/ReadVariableOp�
while/simple_rnn_cell_3/BiasAddBiasAdd(while/simple_rnn_cell_3/MatMul:product:06while/simple_rnn_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2!
while/simple_rnn_cell_3/BiasAdd�
/while/simple_rnn_cell_3/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_3_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype021
/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp�
 while/simple_rnn_cell_3/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2"
 while/simple_rnn_cell_3/MatMul_1�
while/simple_rnn_cell_3/addAddV2(while/simple_rnn_cell_3/BiasAdd:output:0*while/simple_rnn_cell_3/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2
while/simple_rnn_cell_3/add�
while/simple_rnn_cell_3/TanhTanhwhile/simple_rnn_cell_3/add:z:0*
T0*'
_output_shapes
:��������� 2
while/simple_rnn_cell_3/Tanh�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_3/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_3/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_3/MatMul/ReadVariableOp0^while/simple_rnn_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_3/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_3/MatMul/ReadVariableOp0^while/simple_rnn_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_3/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_3/MatMul/ReadVariableOp0^while/simple_rnn_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_3/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_3/MatMul/ReadVariableOp0^while/simple_rnn_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity while/simple_rnn_cell_3/Tanh:y:0/^while/simple_rnn_cell_3/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_3/MatMul/ReadVariableOp0^while/simple_rnn_cell_3/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_3_biasadd_readvariableop_resource9while_simple_rnn_cell_3_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_3_matmul_1_readvariableop_resource:while_simple_rnn_cell_3_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_3_matmul_readvariableop_resource8while_simple_rnn_cell_3_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :��������� : : :::2`
.while/simple_rnn_cell_3/BiasAdd/ReadVariableOp.while/simple_rnn_cell_3/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_3/MatMul/ReadVariableOp-while/simple_rnn_cell_3/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�
�
M__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_102584

inputs
states_0"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2	
BiasAdd�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2

MatMul_1k
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2
addO
TanhTanhadd:z:0*
T0*'
_output_shapes
:��������� 2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2

Identity�

Identity_1IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:��������� :��������� :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states/0
�
�
while_cond_102186
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_102186___redundant_placeholder04
0while_while_cond_102186___redundant_placeholder14
0while_while_cond_102186___redundant_placeholder24
0while_while_cond_102186___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�<
�
G__inference_simple_rnn_5_layer_call_and_return_conditional_losses_99078

inputs
simple_rnn_cell_5_99003
simple_rnn_cell_5_99005
simple_rnn_cell_5_99007
identity��)simple_rnn_cell_5/StatefulPartitionedCall�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
strided_slice_2�
)simple_rnn_cell_5/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_5_99003simple_rnn_cell_5_99005simple_rnn_cell_5_99007*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:��������� :��������� *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_986412+
)simple_rnn_cell_5/StatefulPartitionedCall�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_5_99003simple_rnn_cell_5_99005simple_rnn_cell_5_99007*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :��������� : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_99015*
condR
while_cond_99014*8
output_shapes'
%: : : : :��������� : : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
transpose_1�
IdentityIdentitystrided_slice_3:output:0*^simple_rnn_cell_5/StatefulPartitionedCall^while*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������ :::2V
)simple_rnn_cell_5/StatefulPartitionedCall)simple_rnn_cell_5/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�
�
while_cond_99272
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_99272___redundant_placeholder03
/while_while_cond_99272___redundant_placeholder13
/while_while_cond_99272___redundant_placeholder23
/while_while_cond_99272___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�
�
while_cond_99666
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_99666___redundant_placeholder03
/while_while_cond_99666___redundant_placeholder13
/while_while_cond_99666___redundant_placeholder23
/while_while_cond_99666___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�	
�
-__inference_sequential_6_layer_call_fn_100033
lambda_1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllambda_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_sequential_6_layer_call_and_return_conditional_losses_1000042
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*c
_input_shapesR
P:������������������:::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
0
_output_shapes
:������������������
(
_user_specified_namelambda_1_input
�	
�
2__inference_simple_rnn_cell_4_layer_call_fn_102550

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:��������� :��������� *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_981292
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:��������� 2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:��������� 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:��������� :��������� :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states/0
�	
�
D__inference_dense_22_layer_call_and_return_conditional_losses_102398

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
while_cond_102052
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_102052___redundant_placeholder04
0while_while_cond_102052___redundant_placeholder14
0while_while_cond_102052___redundant_placeholder24
0while_while_cond_102052___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�	
�
2__inference_simple_rnn_cell_5_layer_call_fn_102598

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:��������� :��������� *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_986242
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:��������� 2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:��������� 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:��������� :��������� :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states/0
�
�
while_cond_101202
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_101202___redundant_placeholder04
0while_while_cond_101202___redundant_placeholder14
0while_while_cond_101202___redundant_placeholder24
0while_while_cond_101202___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�H
�
G__inference_simple_rnn_4_layer_call_and_return_conditional_losses_99480

inputs4
0simple_rnn_cell_4_matmul_readvariableop_resource5
1simple_rnn_cell_4_biasadd_readvariableop_resource6
2simple_rnn_cell_4_matmul_1_readvariableop_resource
identity��(simple_rnn_cell_4/BiasAdd/ReadVariableOp�'simple_rnn_cell_4/MatMul/ReadVariableOp�)simple_rnn_cell_4/MatMul_1/ReadVariableOp�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
strided_slice_2�
'simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_4_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02)
'simple_rnn_cell_4/MatMul/ReadVariableOp�
simple_rnn_cell_4/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_4/MatMul�
(simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(simple_rnn_cell_4/BiasAdd/ReadVariableOp�
simple_rnn_cell_4/BiasAddBiasAdd"simple_rnn_cell_4/MatMul:product:00simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_4/BiasAdd�
)simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_4_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype02+
)simple_rnn_cell_4/MatMul_1/ReadVariableOp�
simple_rnn_cell_4/MatMul_1MatMulzeros:output:01simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_4/MatMul_1�
simple_rnn_cell_4/addAddV2"simple_rnn_cell_4/BiasAdd:output:0$simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_4/add�
simple_rnn_cell_4/TanhTanhsimple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_4/Tanh�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_4_matmul_readvariableop_resource1simple_rnn_cell_4_biasadd_readvariableop_resource2simple_rnn_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :��������� : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_99414*
condR
while_cond_99413*8
output_shapes'
%: : : : :��������� : : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
transpose_1�
IdentityIdentitytranspose_1:y:0)^simple_rnn_cell_4/BiasAdd/ReadVariableOp(^simple_rnn_cell_4/MatMul/ReadVariableOp*^simple_rnn_cell_4/MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :������������������ 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������ :::2T
(simple_rnn_cell_4/BiasAdd/ReadVariableOp(simple_rnn_cell_4/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_4/MatMul/ReadVariableOp'simple_rnn_cell_4/MatMul/ReadVariableOp2V
)simple_rnn_cell_4/MatMul_1/ReadVariableOp)simple_rnn_cell_4/MatMul_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�3
�
while_body_101203
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_3_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_3_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_3_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_3_matmul_readvariableop_resource;
7while_simple_rnn_cell_3_biasadd_readvariableop_resource<
8while_simple_rnn_cell_3_matmul_1_readvariableop_resource��.while/simple_rnn_cell_3/BiasAdd/ReadVariableOp�-while/simple_rnn_cell_3/MatMul/ReadVariableOp�/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
-while/simple_rnn_cell_3/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_3_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype02/
-while/simple_rnn_cell_3/MatMul/ReadVariableOp�
while/simple_rnn_cell_3/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2 
while/simple_rnn_cell_3/MatMul�
.while/simple_rnn_cell_3/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_3_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype020
.while/simple_rnn_cell_3/BiasAdd/ReadVariableOp�
while/simple_rnn_cell_3/BiasAddBiasAdd(while/simple_rnn_cell_3/MatMul:product:06while/simple_rnn_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2!
while/simple_rnn_cell_3/BiasAdd�
/while/simple_rnn_cell_3/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_3_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype021
/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp�
 while/simple_rnn_cell_3/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2"
 while/simple_rnn_cell_3/MatMul_1�
while/simple_rnn_cell_3/addAddV2(while/simple_rnn_cell_3/BiasAdd:output:0*while/simple_rnn_cell_3/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2
while/simple_rnn_cell_3/add�
while/simple_rnn_cell_3/TanhTanhwhile/simple_rnn_cell_3/add:z:0*
T0*'
_output_shapes
:��������� 2
while/simple_rnn_cell_3/Tanh�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_3/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_3/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_3/MatMul/ReadVariableOp0^while/simple_rnn_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_3/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_3/MatMul/ReadVariableOp0^while/simple_rnn_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_3/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_3/MatMul/ReadVariableOp0^while/simple_rnn_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_3/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_3/MatMul/ReadVariableOp0^while/simple_rnn_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity while/simple_rnn_cell_3/Tanh:y:0/^while/simple_rnn_cell_3/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_3/MatMul/ReadVariableOp0^while/simple_rnn_cell_3/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_3_biasadd_readvariableop_resource9while_simple_rnn_cell_3_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_3_matmul_1_readvariableop_resource:while_simple_rnn_cell_3_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_3_matmul_readvariableop_resource8while_simple_rnn_cell_3_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :��������� : : :::2`
.while/simple_rnn_cell_3/BiasAdd/ReadVariableOp.while/simple_rnn_cell_3/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_3/MatMul/ReadVariableOp-while/simple_rnn_cell_3/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�
�
M__inference_simple_rnn_cell_3_layer_call_and_return_conditional_losses_102460

inputs
states_0"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2	
BiasAdd�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2

MatMul_1k
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2
addO
TanhTanhadd:z:0*
T0*'
_output_shapes
:��������� 2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2

Identity�

Identity_1IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:���������:��������� :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states/0
؝
�
H__inference_sequential_6_layer_call_and_return_conditional_losses_100484

inputsA
=simple_rnn_3_simple_rnn_cell_3_matmul_readvariableop_resourceB
>simple_rnn_3_simple_rnn_cell_3_biasadd_readvariableop_resourceC
?simple_rnn_3_simple_rnn_cell_3_matmul_1_readvariableop_resourceA
=simple_rnn_4_simple_rnn_cell_4_matmul_readvariableop_resourceB
>simple_rnn_4_simple_rnn_cell_4_biasadd_readvariableop_resourceC
?simple_rnn_4_simple_rnn_cell_4_matmul_1_readvariableop_resourceA
=simple_rnn_5_simple_rnn_cell_5_matmul_readvariableop_resourceB
>simple_rnn_5_simple_rnn_cell_5_biasadd_readvariableop_resourceC
?simple_rnn_5_simple_rnn_cell_5_matmul_1_readvariableop_resource+
'dense_22_matmul_readvariableop_resource,
(dense_22_biasadd_readvariableop_resource+
'dense_23_matmul_readvariableop_resource,
(dense_23_biasadd_readvariableop_resource
identity��dense_22/BiasAdd/ReadVariableOp�dense_22/MatMul/ReadVariableOp�dense_23/BiasAdd/ReadVariableOp�dense_23/MatMul/ReadVariableOp�5simple_rnn_3/simple_rnn_cell_3/BiasAdd/ReadVariableOp�4simple_rnn_3/simple_rnn_cell_3/MatMul/ReadVariableOp�6simple_rnn_3/simple_rnn_cell_3/MatMul_1/ReadVariableOp�simple_rnn_3/while�5simple_rnn_4/simple_rnn_cell_4/BiasAdd/ReadVariableOp�4simple_rnn_4/simple_rnn_cell_4/MatMul/ReadVariableOp�6simple_rnn_4/simple_rnn_cell_4/MatMul_1/ReadVariableOp�simple_rnn_4/while�5simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOp�4simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOp�6simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp�simple_rnn_5/while}
lambda_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
lambda_1/ExpandDims/dim�
lambda_1/ExpandDims
ExpandDimsinputs lambda_1/ExpandDims/dim:output:0*
T0*4
_output_shapes"
 :������������������2
lambda_1/ExpandDimst
simple_rnn_3/ShapeShapelambda_1/ExpandDims:output:0*
T0*
_output_shapes
:2
simple_rnn_3/Shape�
 simple_rnn_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 simple_rnn_3/strided_slice/stack�
"simple_rnn_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_3/strided_slice/stack_1�
"simple_rnn_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_3/strided_slice/stack_2�
simple_rnn_3/strided_sliceStridedSlicesimple_rnn_3/Shape:output:0)simple_rnn_3/strided_slice/stack:output:0+simple_rnn_3/strided_slice/stack_1:output:0+simple_rnn_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_3/strided_slicev
simple_rnn_3/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn_3/zeros/mul/y�
simple_rnn_3/zeros/mulMul#simple_rnn_3/strided_slice:output:0!simple_rnn_3/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_3/zeros/muly
simple_rnn_3/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
simple_rnn_3/zeros/Less/y�
simple_rnn_3/zeros/LessLesssimple_rnn_3/zeros/mul:z:0"simple_rnn_3/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_3/zeros/Less|
simple_rnn_3/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn_3/zeros/packed/1�
simple_rnn_3/zeros/packedPack#simple_rnn_3/strided_slice:output:0$simple_rnn_3/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
simple_rnn_3/zeros/packedy
simple_rnn_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
simple_rnn_3/zeros/Const�
simple_rnn_3/zerosFill"simple_rnn_3/zeros/packed:output:0!simple_rnn_3/zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_3/zeros�
simple_rnn_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_3/transpose/perm�
simple_rnn_3/transpose	Transposelambda_1/ExpandDims:output:0$simple_rnn_3/transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
simple_rnn_3/transposev
simple_rnn_3/Shape_1Shapesimple_rnn_3/transpose:y:0*
T0*
_output_shapes
:2
simple_rnn_3/Shape_1�
"simple_rnn_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn_3/strided_slice_1/stack�
$simple_rnn_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_3/strided_slice_1/stack_1�
$simple_rnn_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_3/strided_slice_1/stack_2�
simple_rnn_3/strided_slice_1StridedSlicesimple_rnn_3/Shape_1:output:0+simple_rnn_3/strided_slice_1/stack:output:0-simple_rnn_3/strided_slice_1/stack_1:output:0-simple_rnn_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_3/strided_slice_1�
(simple_rnn_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2*
(simple_rnn_3/TensorArrayV2/element_shape�
simple_rnn_3/TensorArrayV2TensorListReserve1simple_rnn_3/TensorArrayV2/element_shape:output:0%simple_rnn_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_3/TensorArrayV2�
Bsimple_rnn_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2D
Bsimple_rnn_3/TensorArrayUnstack/TensorListFromTensor/element_shape�
4simple_rnn_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_3/transpose:y:0Ksimple_rnn_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type026
4simple_rnn_3/TensorArrayUnstack/TensorListFromTensor�
"simple_rnn_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn_3/strided_slice_2/stack�
$simple_rnn_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_3/strided_slice_2/stack_1�
$simple_rnn_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_3/strided_slice_2/stack_2�
simple_rnn_3/strided_slice_2StridedSlicesimple_rnn_3/transpose:y:0+simple_rnn_3/strided_slice_2/stack:output:0-simple_rnn_3/strided_slice_2/stack_1:output:0-simple_rnn_3/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
simple_rnn_3/strided_slice_2�
4simple_rnn_3/simple_rnn_cell_3/MatMul/ReadVariableOpReadVariableOp=simple_rnn_3_simple_rnn_cell_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype026
4simple_rnn_3/simple_rnn_cell_3/MatMul/ReadVariableOp�
%simple_rnn_3/simple_rnn_cell_3/MatMulMatMul%simple_rnn_3/strided_slice_2:output:0<simple_rnn_3/simple_rnn_cell_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2'
%simple_rnn_3/simple_rnn_cell_3/MatMul�
5simple_rnn_3/simple_rnn_cell_3/BiasAdd/ReadVariableOpReadVariableOp>simple_rnn_3_simple_rnn_cell_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype027
5simple_rnn_3/simple_rnn_cell_3/BiasAdd/ReadVariableOp�
&simple_rnn_3/simple_rnn_cell_3/BiasAddBiasAdd/simple_rnn_3/simple_rnn_cell_3/MatMul:product:0=simple_rnn_3/simple_rnn_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2(
&simple_rnn_3/simple_rnn_cell_3/BiasAdd�
6simple_rnn_3/simple_rnn_cell_3/MatMul_1/ReadVariableOpReadVariableOp?simple_rnn_3_simple_rnn_cell_3_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype028
6simple_rnn_3/simple_rnn_cell_3/MatMul_1/ReadVariableOp�
'simple_rnn_3/simple_rnn_cell_3/MatMul_1MatMulsimple_rnn_3/zeros:output:0>simple_rnn_3/simple_rnn_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2)
'simple_rnn_3/simple_rnn_cell_3/MatMul_1�
"simple_rnn_3/simple_rnn_cell_3/addAddV2/simple_rnn_3/simple_rnn_cell_3/BiasAdd:output:01simple_rnn_3/simple_rnn_cell_3/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2$
"simple_rnn_3/simple_rnn_cell_3/add�
#simple_rnn_3/simple_rnn_cell_3/TanhTanh&simple_rnn_3/simple_rnn_cell_3/add:z:0*
T0*'
_output_shapes
:��������� 2%
#simple_rnn_3/simple_rnn_cell_3/Tanh�
*simple_rnn_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2,
*simple_rnn_3/TensorArrayV2_1/element_shape�
simple_rnn_3/TensorArrayV2_1TensorListReserve3simple_rnn_3/TensorArrayV2_1/element_shape:output:0%simple_rnn_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_3/TensorArrayV2_1h
simple_rnn_3/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn_3/time�
%simple_rnn_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%simple_rnn_3/while/maximum_iterations�
simple_rnn_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2!
simple_rnn_3/while/loop_counter�
simple_rnn_3/whileWhile(simple_rnn_3/while/loop_counter:output:0.simple_rnn_3/while/maximum_iterations:output:0simple_rnn_3/time:output:0%simple_rnn_3/TensorArrayV2_1:handle:0simple_rnn_3/zeros:output:0%simple_rnn_3/strided_slice_1:output:0Dsimple_rnn_3/TensorArrayUnstack/TensorListFromTensor:output_handle:0=simple_rnn_3_simple_rnn_cell_3_matmul_readvariableop_resource>simple_rnn_3_simple_rnn_cell_3_biasadd_readvariableop_resource?simple_rnn_3_simple_rnn_cell_3_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :��������� : : : : : *%
_read_only_resource_inputs
	**
body"R 
simple_rnn_3_while_body_100189**
cond"R 
simple_rnn_3_while_cond_100188*8
output_shapes'
%: : : : :��������� : : : : : *
parallel_iterations 2
simple_rnn_3/while�
=simple_rnn_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2?
=simple_rnn_3/TensorArrayV2Stack/TensorListStack/element_shape�
/simple_rnn_3/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_3/while:output:3Fsimple_rnn_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������ *
element_dtype021
/simple_rnn_3/TensorArrayV2Stack/TensorListStack�
"simple_rnn_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2$
"simple_rnn_3/strided_slice_3/stack�
$simple_rnn_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$simple_rnn_3/strided_slice_3/stack_1�
$simple_rnn_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_3/strided_slice_3/stack_2�
simple_rnn_3/strided_slice_3StridedSlice8simple_rnn_3/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_3/strided_slice_3/stack:output:0-simple_rnn_3/strided_slice_3/stack_1:output:0-simple_rnn_3/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
simple_rnn_3/strided_slice_3�
simple_rnn_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_3/transpose_1/perm�
simple_rnn_3/transpose_1	Transpose8simple_rnn_3/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_3/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
simple_rnn_3/transpose_1t
simple_rnn_4/ShapeShapesimple_rnn_3/transpose_1:y:0*
T0*
_output_shapes
:2
simple_rnn_4/Shape�
 simple_rnn_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 simple_rnn_4/strided_slice/stack�
"simple_rnn_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_4/strided_slice/stack_1�
"simple_rnn_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_4/strided_slice/stack_2�
simple_rnn_4/strided_sliceStridedSlicesimple_rnn_4/Shape:output:0)simple_rnn_4/strided_slice/stack:output:0+simple_rnn_4/strided_slice/stack_1:output:0+simple_rnn_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_4/strided_slicev
simple_rnn_4/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn_4/zeros/mul/y�
simple_rnn_4/zeros/mulMul#simple_rnn_4/strided_slice:output:0!simple_rnn_4/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_4/zeros/muly
simple_rnn_4/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
simple_rnn_4/zeros/Less/y�
simple_rnn_4/zeros/LessLesssimple_rnn_4/zeros/mul:z:0"simple_rnn_4/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_4/zeros/Less|
simple_rnn_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn_4/zeros/packed/1�
simple_rnn_4/zeros/packedPack#simple_rnn_4/strided_slice:output:0$simple_rnn_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
simple_rnn_4/zeros/packedy
simple_rnn_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
simple_rnn_4/zeros/Const�
simple_rnn_4/zerosFill"simple_rnn_4/zeros/packed:output:0!simple_rnn_4/zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_4/zeros�
simple_rnn_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_4/transpose/perm�
simple_rnn_4/transpose	Transposesimple_rnn_3/transpose_1:y:0$simple_rnn_4/transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
simple_rnn_4/transposev
simple_rnn_4/Shape_1Shapesimple_rnn_4/transpose:y:0*
T0*
_output_shapes
:2
simple_rnn_4/Shape_1�
"simple_rnn_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn_4/strided_slice_1/stack�
$simple_rnn_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_4/strided_slice_1/stack_1�
$simple_rnn_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_4/strided_slice_1/stack_2�
simple_rnn_4/strided_slice_1StridedSlicesimple_rnn_4/Shape_1:output:0+simple_rnn_4/strided_slice_1/stack:output:0-simple_rnn_4/strided_slice_1/stack_1:output:0-simple_rnn_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_4/strided_slice_1�
(simple_rnn_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2*
(simple_rnn_4/TensorArrayV2/element_shape�
simple_rnn_4/TensorArrayV2TensorListReserve1simple_rnn_4/TensorArrayV2/element_shape:output:0%simple_rnn_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_4/TensorArrayV2�
Bsimple_rnn_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2D
Bsimple_rnn_4/TensorArrayUnstack/TensorListFromTensor/element_shape�
4simple_rnn_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_4/transpose:y:0Ksimple_rnn_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type026
4simple_rnn_4/TensorArrayUnstack/TensorListFromTensor�
"simple_rnn_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn_4/strided_slice_2/stack�
$simple_rnn_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_4/strided_slice_2/stack_1�
$simple_rnn_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_4/strided_slice_2/stack_2�
simple_rnn_4/strided_slice_2StridedSlicesimple_rnn_4/transpose:y:0+simple_rnn_4/strided_slice_2/stack:output:0-simple_rnn_4/strided_slice_2/stack_1:output:0-simple_rnn_4/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
simple_rnn_4/strided_slice_2�
4simple_rnn_4/simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp=simple_rnn_4_simple_rnn_cell_4_matmul_readvariableop_resource*
_output_shapes

:  *
dtype026
4simple_rnn_4/simple_rnn_cell_4/MatMul/ReadVariableOp�
%simple_rnn_4/simple_rnn_cell_4/MatMulMatMul%simple_rnn_4/strided_slice_2:output:0<simple_rnn_4/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2'
%simple_rnn_4/simple_rnn_cell_4/MatMul�
5simple_rnn_4/simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp>simple_rnn_4_simple_rnn_cell_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype027
5simple_rnn_4/simple_rnn_cell_4/BiasAdd/ReadVariableOp�
&simple_rnn_4/simple_rnn_cell_4/BiasAddBiasAdd/simple_rnn_4/simple_rnn_cell_4/MatMul:product:0=simple_rnn_4/simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2(
&simple_rnn_4/simple_rnn_cell_4/BiasAdd�
6simple_rnn_4/simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp?simple_rnn_4_simple_rnn_cell_4_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype028
6simple_rnn_4/simple_rnn_cell_4/MatMul_1/ReadVariableOp�
'simple_rnn_4/simple_rnn_cell_4/MatMul_1MatMulsimple_rnn_4/zeros:output:0>simple_rnn_4/simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2)
'simple_rnn_4/simple_rnn_cell_4/MatMul_1�
"simple_rnn_4/simple_rnn_cell_4/addAddV2/simple_rnn_4/simple_rnn_cell_4/BiasAdd:output:01simple_rnn_4/simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2$
"simple_rnn_4/simple_rnn_cell_4/add�
#simple_rnn_4/simple_rnn_cell_4/TanhTanh&simple_rnn_4/simple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:��������� 2%
#simple_rnn_4/simple_rnn_cell_4/Tanh�
*simple_rnn_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2,
*simple_rnn_4/TensorArrayV2_1/element_shape�
simple_rnn_4/TensorArrayV2_1TensorListReserve3simple_rnn_4/TensorArrayV2_1/element_shape:output:0%simple_rnn_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_4/TensorArrayV2_1h
simple_rnn_4/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn_4/time�
%simple_rnn_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%simple_rnn_4/while/maximum_iterations�
simple_rnn_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2!
simple_rnn_4/while/loop_counter�
simple_rnn_4/whileWhile(simple_rnn_4/while/loop_counter:output:0.simple_rnn_4/while/maximum_iterations:output:0simple_rnn_4/time:output:0%simple_rnn_4/TensorArrayV2_1:handle:0simple_rnn_4/zeros:output:0%simple_rnn_4/strided_slice_1:output:0Dsimple_rnn_4/TensorArrayUnstack/TensorListFromTensor:output_handle:0=simple_rnn_4_simple_rnn_cell_4_matmul_readvariableop_resource>simple_rnn_4_simple_rnn_cell_4_biasadd_readvariableop_resource?simple_rnn_4_simple_rnn_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :��������� : : : : : *%
_read_only_resource_inputs
	**
body"R 
simple_rnn_4_while_body_100297**
cond"R 
simple_rnn_4_while_cond_100296*8
output_shapes'
%: : : : :��������� : : : : : *
parallel_iterations 2
simple_rnn_4/while�
=simple_rnn_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2?
=simple_rnn_4/TensorArrayV2Stack/TensorListStack/element_shape�
/simple_rnn_4/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_4/while:output:3Fsimple_rnn_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������ *
element_dtype021
/simple_rnn_4/TensorArrayV2Stack/TensorListStack�
"simple_rnn_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2$
"simple_rnn_4/strided_slice_3/stack�
$simple_rnn_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$simple_rnn_4/strided_slice_3/stack_1�
$simple_rnn_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_4/strided_slice_3/stack_2�
simple_rnn_4/strided_slice_3StridedSlice8simple_rnn_4/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_4/strided_slice_3/stack:output:0-simple_rnn_4/strided_slice_3/stack_1:output:0-simple_rnn_4/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
simple_rnn_4/strided_slice_3�
simple_rnn_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_4/transpose_1/perm�
simple_rnn_4/transpose_1	Transpose8simple_rnn_4/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_4/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
simple_rnn_4/transpose_1t
simple_rnn_5/ShapeShapesimple_rnn_4/transpose_1:y:0*
T0*
_output_shapes
:2
simple_rnn_5/Shape�
 simple_rnn_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 simple_rnn_5/strided_slice/stack�
"simple_rnn_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_5/strided_slice/stack_1�
"simple_rnn_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_5/strided_slice/stack_2�
simple_rnn_5/strided_sliceStridedSlicesimple_rnn_5/Shape:output:0)simple_rnn_5/strided_slice/stack:output:0+simple_rnn_5/strided_slice/stack_1:output:0+simple_rnn_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_5/strided_slicev
simple_rnn_5/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn_5/zeros/mul/y�
simple_rnn_5/zeros/mulMul#simple_rnn_5/strided_slice:output:0!simple_rnn_5/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_5/zeros/muly
simple_rnn_5/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
simple_rnn_5/zeros/Less/y�
simple_rnn_5/zeros/LessLesssimple_rnn_5/zeros/mul:z:0"simple_rnn_5/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_5/zeros/Less|
simple_rnn_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn_5/zeros/packed/1�
simple_rnn_5/zeros/packedPack#simple_rnn_5/strided_slice:output:0$simple_rnn_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
simple_rnn_5/zeros/packedy
simple_rnn_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
simple_rnn_5/zeros/Const�
simple_rnn_5/zerosFill"simple_rnn_5/zeros/packed:output:0!simple_rnn_5/zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_5/zeros�
simple_rnn_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_5/transpose/perm�
simple_rnn_5/transpose	Transposesimple_rnn_4/transpose_1:y:0$simple_rnn_5/transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
simple_rnn_5/transposev
simple_rnn_5/Shape_1Shapesimple_rnn_5/transpose:y:0*
T0*
_output_shapes
:2
simple_rnn_5/Shape_1�
"simple_rnn_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn_5/strided_slice_1/stack�
$simple_rnn_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_5/strided_slice_1/stack_1�
$simple_rnn_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_5/strided_slice_1/stack_2�
simple_rnn_5/strided_slice_1StridedSlicesimple_rnn_5/Shape_1:output:0+simple_rnn_5/strided_slice_1/stack:output:0-simple_rnn_5/strided_slice_1/stack_1:output:0-simple_rnn_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_5/strided_slice_1�
(simple_rnn_5/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2*
(simple_rnn_5/TensorArrayV2/element_shape�
simple_rnn_5/TensorArrayV2TensorListReserve1simple_rnn_5/TensorArrayV2/element_shape:output:0%simple_rnn_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_5/TensorArrayV2�
Bsimple_rnn_5/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2D
Bsimple_rnn_5/TensorArrayUnstack/TensorListFromTensor/element_shape�
4simple_rnn_5/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_5/transpose:y:0Ksimple_rnn_5/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type026
4simple_rnn_5/TensorArrayUnstack/TensorListFromTensor�
"simple_rnn_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn_5/strided_slice_2/stack�
$simple_rnn_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_5/strided_slice_2/stack_1�
$simple_rnn_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_5/strided_slice_2/stack_2�
simple_rnn_5/strided_slice_2StridedSlicesimple_rnn_5/transpose:y:0+simple_rnn_5/strided_slice_2/stack:output:0-simple_rnn_5/strided_slice_2/stack_1:output:0-simple_rnn_5/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
simple_rnn_5/strided_slice_2�
4simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp=simple_rnn_5_simple_rnn_cell_5_matmul_readvariableop_resource*
_output_shapes

:  *
dtype026
4simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOp�
%simple_rnn_5/simple_rnn_cell_5/MatMulMatMul%simple_rnn_5/strided_slice_2:output:0<simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2'
%simple_rnn_5/simple_rnn_cell_5/MatMul�
5simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp>simple_rnn_5_simple_rnn_cell_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype027
5simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOp�
&simple_rnn_5/simple_rnn_cell_5/BiasAddBiasAdd/simple_rnn_5/simple_rnn_cell_5/MatMul:product:0=simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2(
&simple_rnn_5/simple_rnn_cell_5/BiasAdd�
6simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp?simple_rnn_5_simple_rnn_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype028
6simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp�
'simple_rnn_5/simple_rnn_cell_5/MatMul_1MatMulsimple_rnn_5/zeros:output:0>simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2)
'simple_rnn_5/simple_rnn_cell_5/MatMul_1�
"simple_rnn_5/simple_rnn_cell_5/addAddV2/simple_rnn_5/simple_rnn_cell_5/BiasAdd:output:01simple_rnn_5/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2$
"simple_rnn_5/simple_rnn_cell_5/add�
#simple_rnn_5/simple_rnn_cell_5/TanhTanh&simple_rnn_5/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:��������� 2%
#simple_rnn_5/simple_rnn_cell_5/Tanh�
*simple_rnn_5/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2,
*simple_rnn_5/TensorArrayV2_1/element_shape�
simple_rnn_5/TensorArrayV2_1TensorListReserve3simple_rnn_5/TensorArrayV2_1/element_shape:output:0%simple_rnn_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_5/TensorArrayV2_1h
simple_rnn_5/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn_5/time�
%simple_rnn_5/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%simple_rnn_5/while/maximum_iterations�
simple_rnn_5/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2!
simple_rnn_5/while/loop_counter�
simple_rnn_5/whileWhile(simple_rnn_5/while/loop_counter:output:0.simple_rnn_5/while/maximum_iterations:output:0simple_rnn_5/time:output:0%simple_rnn_5/TensorArrayV2_1:handle:0simple_rnn_5/zeros:output:0%simple_rnn_5/strided_slice_1:output:0Dsimple_rnn_5/TensorArrayUnstack/TensorListFromTensor:output_handle:0=simple_rnn_5_simple_rnn_cell_5_matmul_readvariableop_resource>simple_rnn_5_simple_rnn_cell_5_biasadd_readvariableop_resource?simple_rnn_5_simple_rnn_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :��������� : : : : : *%
_read_only_resource_inputs
	**
body"R 
simple_rnn_5_while_body_100405**
cond"R 
simple_rnn_5_while_cond_100404*8
output_shapes'
%: : : : :��������� : : : : : *
parallel_iterations 2
simple_rnn_5/while�
=simple_rnn_5/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2?
=simple_rnn_5/TensorArrayV2Stack/TensorListStack/element_shape�
/simple_rnn_5/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_5/while:output:3Fsimple_rnn_5/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������ *
element_dtype021
/simple_rnn_5/TensorArrayV2Stack/TensorListStack�
"simple_rnn_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2$
"simple_rnn_5/strided_slice_3/stack�
$simple_rnn_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$simple_rnn_5/strided_slice_3/stack_1�
$simple_rnn_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_5/strided_slice_3/stack_2�
simple_rnn_5/strided_slice_3StridedSlice8simple_rnn_5/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_5/strided_slice_3/stack:output:0-simple_rnn_5/strided_slice_3/stack_1:output:0-simple_rnn_5/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
simple_rnn_5/strided_slice_3�
simple_rnn_5/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_5/transpose_1/perm�
simple_rnn_5/transpose_1	Transpose8simple_rnn_5/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_5/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
simple_rnn_5/transpose_1�
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_22/MatMul/ReadVariableOp�
dense_22/MatMulMatMul%simple_rnn_5/strided_slice_3:output:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_22/MatMul�
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_22/BiasAdd/ReadVariableOp�
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_22/BiasAdds
dense_22/ReluReludense_22/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_22/Relu�
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_23/MatMul/ReadVariableOp�
dense_23/MatMulMatMuldense_22/Relu:activations:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_23/MatMul�
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_23/BiasAdd/ReadVariableOp�
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_23/BiasAdd�
IdentityIdentitydense_23/BiasAdd:output:0 ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp6^simple_rnn_3/simple_rnn_cell_3/BiasAdd/ReadVariableOp5^simple_rnn_3/simple_rnn_cell_3/MatMul/ReadVariableOp7^simple_rnn_3/simple_rnn_cell_3/MatMul_1/ReadVariableOp^simple_rnn_3/while6^simple_rnn_4/simple_rnn_cell_4/BiasAdd/ReadVariableOp5^simple_rnn_4/simple_rnn_cell_4/MatMul/ReadVariableOp7^simple_rnn_4/simple_rnn_cell_4/MatMul_1/ReadVariableOp^simple_rnn_4/while6^simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOp5^simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOp7^simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp^simple_rnn_5/while*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*c
_input_shapesR
P:������������������:::::::::::::2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2n
5simple_rnn_3/simple_rnn_cell_3/BiasAdd/ReadVariableOp5simple_rnn_3/simple_rnn_cell_3/BiasAdd/ReadVariableOp2l
4simple_rnn_3/simple_rnn_cell_3/MatMul/ReadVariableOp4simple_rnn_3/simple_rnn_cell_3/MatMul/ReadVariableOp2p
6simple_rnn_3/simple_rnn_cell_3/MatMul_1/ReadVariableOp6simple_rnn_3/simple_rnn_cell_3/MatMul_1/ReadVariableOp2(
simple_rnn_3/whilesimple_rnn_3/while2n
5simple_rnn_4/simple_rnn_cell_4/BiasAdd/ReadVariableOp5simple_rnn_4/simple_rnn_cell_4/BiasAdd/ReadVariableOp2l
4simple_rnn_4/simple_rnn_cell_4/MatMul/ReadVariableOp4simple_rnn_4/simple_rnn_cell_4/MatMul/ReadVariableOp2p
6simple_rnn_4/simple_rnn_cell_4/MatMul_1/ReadVariableOp6simple_rnn_4/simple_rnn_cell_4/MatMul_1/ReadVariableOp2(
simple_rnn_4/whilesimple_rnn_4/while2n
5simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOp5simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOp2l
4simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOp4simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOp2p
6simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp6simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp2(
simple_rnn_5/whilesimple_rnn_5/while:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
�
L__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_98112

inputs

states"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2	
BiasAdd�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul_1/ReadVariableOpy
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2

MatMul_1k
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2
addO
TanhTanhadd:z:0*
T0*'
_output_shapes
:��������� 2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2

Identity�

Identity_1IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:��������� :��������� :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_namestates
�3
�
while_body_101069
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_3_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_3_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_3_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_3_matmul_readvariableop_resource;
7while_simple_rnn_cell_3_biasadd_readvariableop_resource<
8while_simple_rnn_cell_3_matmul_1_readvariableop_resource��.while/simple_rnn_cell_3/BiasAdd/ReadVariableOp�-while/simple_rnn_cell_3/MatMul/ReadVariableOp�/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
-while/simple_rnn_cell_3/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_3_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype02/
-while/simple_rnn_cell_3/MatMul/ReadVariableOp�
while/simple_rnn_cell_3/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2 
while/simple_rnn_cell_3/MatMul�
.while/simple_rnn_cell_3/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_3_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype020
.while/simple_rnn_cell_3/BiasAdd/ReadVariableOp�
while/simple_rnn_cell_3/BiasAddBiasAdd(while/simple_rnn_cell_3/MatMul:product:06while/simple_rnn_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2!
while/simple_rnn_cell_3/BiasAdd�
/while/simple_rnn_cell_3/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_3_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype021
/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp�
 while/simple_rnn_cell_3/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2"
 while/simple_rnn_cell_3/MatMul_1�
while/simple_rnn_cell_3/addAddV2(while/simple_rnn_cell_3/BiasAdd:output:0*while/simple_rnn_cell_3/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2
while/simple_rnn_cell_3/add�
while/simple_rnn_cell_3/TanhTanhwhile/simple_rnn_cell_3/add:z:0*
T0*'
_output_shapes
:��������� 2
while/simple_rnn_cell_3/Tanh�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_3/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_3/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_3/MatMul/ReadVariableOp0^while/simple_rnn_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_3/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_3/MatMul/ReadVariableOp0^while/simple_rnn_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_3/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_3/MatMul/ReadVariableOp0^while/simple_rnn_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_3/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_3/MatMul/ReadVariableOp0^while/simple_rnn_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity while/simple_rnn_cell_3/Tanh:y:0/^while/simple_rnn_cell_3/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_3/MatMul/ReadVariableOp0^while/simple_rnn_cell_3/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_3_biasadd_readvariableop_resource9while_simple_rnn_cell_3_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_3_matmul_1_readvariableop_resource:while_simple_rnn_cell_3_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_3_matmul_readvariableop_resource8while_simple_rnn_cell_3_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :��������� : : :::2`
.while/simple_rnn_cell_3/BiasAdd/ReadVariableOp.while/simple_rnn_cell_3/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_3/MatMul/ReadVariableOp-while/simple_rnn_cell_3/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�
�
-__inference_simple_rnn_4_layer_call_fn_101649

inputs
unknown
	unknown_0
	unknown_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_simple_rnn_4_layer_call_and_return_conditional_losses_995922
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :������������������ 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������ :::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�#
�
while_body_97874
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0#
while_simple_rnn_cell_3_97896_0#
while_simple_rnn_cell_3_97898_0#
while_simple_rnn_cell_3_97900_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor!
while_simple_rnn_cell_3_97896!
while_simple_rnn_cell_3_97898!
while_simple_rnn_cell_3_97900��/while/simple_rnn_cell_3/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
/while/simple_rnn_cell_3/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_simple_rnn_cell_3_97896_0while_simple_rnn_cell_3_97898_0while_simple_rnn_cell_3_97900_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:��������� :��������� *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_simple_rnn_cell_3_layer_call_and_return_conditional_losses_9760021
/while/simple_rnn_cell_3/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder8while/simple_rnn_cell_3/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:00^while/simple_rnn_cell_3/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations0^while/simple_rnn_cell_3/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:00^while/simple_rnn_cell_3/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/simple_rnn_cell_3/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity8while/simple_rnn_cell_3/StatefulPartitionedCall:output:10^while/simple_rnn_cell_3/StatefulPartitionedCall*
T0*'
_output_shapes
:��������� 2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"@
while_simple_rnn_cell_3_97896while_simple_rnn_cell_3_97896_0"@
while_simple_rnn_cell_3_97898while_simple_rnn_cell_3_97898_0"@
while_simple_rnn_cell_3_97900while_simple_rnn_cell_3_97900_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :��������� : : :::2b
/while/simple_rnn_cell_3/StatefulPartitionedCall/while/simple_rnn_cell_3/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�H
�
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_102007
inputs_04
0simple_rnn_cell_5_matmul_readvariableop_resource5
1simple_rnn_cell_5_biasadd_readvariableop_resource6
2simple_rnn_cell_5_matmul_1_readvariableop_resource
identity��(simple_rnn_cell_5/BiasAdd/ReadVariableOp�'simple_rnn_cell_5/MatMul/ReadVariableOp�)simple_rnn_cell_5/MatMul_1/ReadVariableOp�whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
strided_slice_2�
'simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_5_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02)
'simple_rnn_cell_5/MatMul/ReadVariableOp�
simple_rnn_cell_5/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_5/MatMul�
(simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(simple_rnn_cell_5/BiasAdd/ReadVariableOp�
simple_rnn_cell_5/BiasAddBiasAdd"simple_rnn_cell_5/MatMul:product:00simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_5/BiasAdd�
)simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype02+
)simple_rnn_cell_5/MatMul_1/ReadVariableOp�
simple_rnn_cell_5/MatMul_1MatMulzeros:output:01simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_5/MatMul_1�
simple_rnn_cell_5/addAddV2"simple_rnn_cell_5/BiasAdd:output:0$simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_5/add�
simple_rnn_cell_5/TanhTanhsimple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_5/Tanh�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_5_matmul_readvariableop_resource1simple_rnn_cell_5_biasadd_readvariableop_resource2simple_rnn_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :��������� : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_101941*
condR
while_cond_101940*8
output_shapes'
%: : : : :��������� : : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
transpose_1�
IdentityIdentitystrided_slice_3:output:0)^simple_rnn_cell_5/BiasAdd/ReadVariableOp(^simple_rnn_cell_5/MatMul/ReadVariableOp*^simple_rnn_cell_5/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������ :::2T
(simple_rnn_cell_5/BiasAdd/ReadVariableOp(simple_rnn_cell_5/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_5/MatMul/ReadVariableOp'simple_rnn_cell_5/MatMul/ReadVariableOp2V
)simple_rnn_cell_5/MatMul_1/ReadVariableOp)simple_rnn_cell_5/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������ 
"
_user_specified_name
inputs/0
�
�
M__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_102567

inputs
states_0"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2	
BiasAdd�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2

MatMul_1k
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2
addO
TanhTanhadd:z:0*
T0*'
_output_shapes
:��������� 2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2

Identity�

Identity_1IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:��������� :��������� :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states/0
��
�
 __inference__wrapped_model_97551
lambda_1_inputN
Jsequential_6_simple_rnn_3_simple_rnn_cell_3_matmul_readvariableop_resourceO
Ksequential_6_simple_rnn_3_simple_rnn_cell_3_biasadd_readvariableop_resourceP
Lsequential_6_simple_rnn_3_simple_rnn_cell_3_matmul_1_readvariableop_resourceN
Jsequential_6_simple_rnn_4_simple_rnn_cell_4_matmul_readvariableop_resourceO
Ksequential_6_simple_rnn_4_simple_rnn_cell_4_biasadd_readvariableop_resourceP
Lsequential_6_simple_rnn_4_simple_rnn_cell_4_matmul_1_readvariableop_resourceN
Jsequential_6_simple_rnn_5_simple_rnn_cell_5_matmul_readvariableop_resourceO
Ksequential_6_simple_rnn_5_simple_rnn_cell_5_biasadd_readvariableop_resourceP
Lsequential_6_simple_rnn_5_simple_rnn_cell_5_matmul_1_readvariableop_resource8
4sequential_6_dense_22_matmul_readvariableop_resource9
5sequential_6_dense_22_biasadd_readvariableop_resource8
4sequential_6_dense_23_matmul_readvariableop_resource9
5sequential_6_dense_23_biasadd_readvariableop_resource
identity��,sequential_6/dense_22/BiasAdd/ReadVariableOp�+sequential_6/dense_22/MatMul/ReadVariableOp�,sequential_6/dense_23/BiasAdd/ReadVariableOp�+sequential_6/dense_23/MatMul/ReadVariableOp�Bsequential_6/simple_rnn_3/simple_rnn_cell_3/BiasAdd/ReadVariableOp�Asequential_6/simple_rnn_3/simple_rnn_cell_3/MatMul/ReadVariableOp�Csequential_6/simple_rnn_3/simple_rnn_cell_3/MatMul_1/ReadVariableOp�sequential_6/simple_rnn_3/while�Bsequential_6/simple_rnn_4/simple_rnn_cell_4/BiasAdd/ReadVariableOp�Asequential_6/simple_rnn_4/simple_rnn_cell_4/MatMul/ReadVariableOp�Csequential_6/simple_rnn_4/simple_rnn_cell_4/MatMul_1/ReadVariableOp�sequential_6/simple_rnn_4/while�Bsequential_6/simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOp�Asequential_6/simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOp�Csequential_6/simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp�sequential_6/simple_rnn_5/while�
$sequential_6/lambda_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2&
$sequential_6/lambda_1/ExpandDims/dim�
 sequential_6/lambda_1/ExpandDims
ExpandDimslambda_1_input-sequential_6/lambda_1/ExpandDims/dim:output:0*
T0*4
_output_shapes"
 :������������������2"
 sequential_6/lambda_1/ExpandDims�
sequential_6/simple_rnn_3/ShapeShape)sequential_6/lambda_1/ExpandDims:output:0*
T0*
_output_shapes
:2!
sequential_6/simple_rnn_3/Shape�
-sequential_6/simple_rnn_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_6/simple_rnn_3/strided_slice/stack�
/sequential_6/simple_rnn_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/sequential_6/simple_rnn_3/strided_slice/stack_1�
/sequential_6/simple_rnn_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/sequential_6/simple_rnn_3/strided_slice/stack_2�
'sequential_6/simple_rnn_3/strided_sliceStridedSlice(sequential_6/simple_rnn_3/Shape:output:06sequential_6/simple_rnn_3/strided_slice/stack:output:08sequential_6/simple_rnn_3/strided_slice/stack_1:output:08sequential_6/simple_rnn_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'sequential_6/simple_rnn_3/strided_slice�
%sequential_6/simple_rnn_3/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2'
%sequential_6/simple_rnn_3/zeros/mul/y�
#sequential_6/simple_rnn_3/zeros/mulMul0sequential_6/simple_rnn_3/strided_slice:output:0.sequential_6/simple_rnn_3/zeros/mul/y:output:0*
T0*
_output_shapes
: 2%
#sequential_6/simple_rnn_3/zeros/mul�
&sequential_6/simple_rnn_3/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2(
&sequential_6/simple_rnn_3/zeros/Less/y�
$sequential_6/simple_rnn_3/zeros/LessLess'sequential_6/simple_rnn_3/zeros/mul:z:0/sequential_6/simple_rnn_3/zeros/Less/y:output:0*
T0*
_output_shapes
: 2&
$sequential_6/simple_rnn_3/zeros/Less�
(sequential_6/simple_rnn_3/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_6/simple_rnn_3/zeros/packed/1�
&sequential_6/simple_rnn_3/zeros/packedPack0sequential_6/simple_rnn_3/strided_slice:output:01sequential_6/simple_rnn_3/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_6/simple_rnn_3/zeros/packed�
%sequential_6/simple_rnn_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%sequential_6/simple_rnn_3/zeros/Const�
sequential_6/simple_rnn_3/zerosFill/sequential_6/simple_rnn_3/zeros/packed:output:0.sequential_6/simple_rnn_3/zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2!
sequential_6/simple_rnn_3/zeros�
(sequential_6/simple_rnn_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(sequential_6/simple_rnn_3/transpose/perm�
#sequential_6/simple_rnn_3/transpose	Transpose)sequential_6/lambda_1/ExpandDims:output:01sequential_6/simple_rnn_3/transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2%
#sequential_6/simple_rnn_3/transpose�
!sequential_6/simple_rnn_3/Shape_1Shape'sequential_6/simple_rnn_3/transpose:y:0*
T0*
_output_shapes
:2#
!sequential_6/simple_rnn_3/Shape_1�
/sequential_6/simple_rnn_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/sequential_6/simple_rnn_3/strided_slice_1/stack�
1sequential_6/simple_rnn_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential_6/simple_rnn_3/strided_slice_1/stack_1�
1sequential_6/simple_rnn_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential_6/simple_rnn_3/strided_slice_1/stack_2�
)sequential_6/simple_rnn_3/strided_slice_1StridedSlice*sequential_6/simple_rnn_3/Shape_1:output:08sequential_6/simple_rnn_3/strided_slice_1/stack:output:0:sequential_6/simple_rnn_3/strided_slice_1/stack_1:output:0:sequential_6/simple_rnn_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)sequential_6/simple_rnn_3/strided_slice_1�
5sequential_6/simple_rnn_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������27
5sequential_6/simple_rnn_3/TensorArrayV2/element_shape�
'sequential_6/simple_rnn_3/TensorArrayV2TensorListReserve>sequential_6/simple_rnn_3/TensorArrayV2/element_shape:output:02sequential_6/simple_rnn_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'sequential_6/simple_rnn_3/TensorArrayV2�
Osequential_6/simple_rnn_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2Q
Osequential_6/simple_rnn_3/TensorArrayUnstack/TensorListFromTensor/element_shape�
Asequential_6/simple_rnn_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor'sequential_6/simple_rnn_3/transpose:y:0Xsequential_6/simple_rnn_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02C
Asequential_6/simple_rnn_3/TensorArrayUnstack/TensorListFromTensor�
/sequential_6/simple_rnn_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/sequential_6/simple_rnn_3/strided_slice_2/stack�
1sequential_6/simple_rnn_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential_6/simple_rnn_3/strided_slice_2/stack_1�
1sequential_6/simple_rnn_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential_6/simple_rnn_3/strided_slice_2/stack_2�
)sequential_6/simple_rnn_3/strided_slice_2StridedSlice'sequential_6/simple_rnn_3/transpose:y:08sequential_6/simple_rnn_3/strided_slice_2/stack:output:0:sequential_6/simple_rnn_3/strided_slice_2/stack_1:output:0:sequential_6/simple_rnn_3/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2+
)sequential_6/simple_rnn_3/strided_slice_2�
Asequential_6/simple_rnn_3/simple_rnn_cell_3/MatMul/ReadVariableOpReadVariableOpJsequential_6_simple_rnn_3_simple_rnn_cell_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02C
Asequential_6/simple_rnn_3/simple_rnn_cell_3/MatMul/ReadVariableOp�
2sequential_6/simple_rnn_3/simple_rnn_cell_3/MatMulMatMul2sequential_6/simple_rnn_3/strided_slice_2:output:0Isequential_6/simple_rnn_3/simple_rnn_cell_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 24
2sequential_6/simple_rnn_3/simple_rnn_cell_3/MatMul�
Bsequential_6/simple_rnn_3/simple_rnn_cell_3/BiasAdd/ReadVariableOpReadVariableOpKsequential_6_simple_rnn_3_simple_rnn_cell_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02D
Bsequential_6/simple_rnn_3/simple_rnn_cell_3/BiasAdd/ReadVariableOp�
3sequential_6/simple_rnn_3/simple_rnn_cell_3/BiasAddBiasAdd<sequential_6/simple_rnn_3/simple_rnn_cell_3/MatMul:product:0Jsequential_6/simple_rnn_3/simple_rnn_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 25
3sequential_6/simple_rnn_3/simple_rnn_cell_3/BiasAdd�
Csequential_6/simple_rnn_3/simple_rnn_cell_3/MatMul_1/ReadVariableOpReadVariableOpLsequential_6_simple_rnn_3_simple_rnn_cell_3_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype02E
Csequential_6/simple_rnn_3/simple_rnn_cell_3/MatMul_1/ReadVariableOp�
4sequential_6/simple_rnn_3/simple_rnn_cell_3/MatMul_1MatMul(sequential_6/simple_rnn_3/zeros:output:0Ksequential_6/simple_rnn_3/simple_rnn_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 26
4sequential_6/simple_rnn_3/simple_rnn_cell_3/MatMul_1�
/sequential_6/simple_rnn_3/simple_rnn_cell_3/addAddV2<sequential_6/simple_rnn_3/simple_rnn_cell_3/BiasAdd:output:0>sequential_6/simple_rnn_3/simple_rnn_cell_3/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 21
/sequential_6/simple_rnn_3/simple_rnn_cell_3/add�
0sequential_6/simple_rnn_3/simple_rnn_cell_3/TanhTanh3sequential_6/simple_rnn_3/simple_rnn_cell_3/add:z:0*
T0*'
_output_shapes
:��������� 22
0sequential_6/simple_rnn_3/simple_rnn_cell_3/Tanh�
7sequential_6/simple_rnn_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    29
7sequential_6/simple_rnn_3/TensorArrayV2_1/element_shape�
)sequential_6/simple_rnn_3/TensorArrayV2_1TensorListReserve@sequential_6/simple_rnn_3/TensorArrayV2_1/element_shape:output:02sequential_6/simple_rnn_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02+
)sequential_6/simple_rnn_3/TensorArrayV2_1�
sequential_6/simple_rnn_3/timeConst*
_output_shapes
: *
dtype0*
value	B : 2 
sequential_6/simple_rnn_3/time�
2sequential_6/simple_rnn_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������24
2sequential_6/simple_rnn_3/while/maximum_iterations�
,sequential_6/simple_rnn_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_6/simple_rnn_3/while/loop_counter�
sequential_6/simple_rnn_3/whileWhile5sequential_6/simple_rnn_3/while/loop_counter:output:0;sequential_6/simple_rnn_3/while/maximum_iterations:output:0'sequential_6/simple_rnn_3/time:output:02sequential_6/simple_rnn_3/TensorArrayV2_1:handle:0(sequential_6/simple_rnn_3/zeros:output:02sequential_6/simple_rnn_3/strided_slice_1:output:0Qsequential_6/simple_rnn_3/TensorArrayUnstack/TensorListFromTensor:output_handle:0Jsequential_6_simple_rnn_3_simple_rnn_cell_3_matmul_readvariableop_resourceKsequential_6_simple_rnn_3_simple_rnn_cell_3_biasadd_readvariableop_resourceLsequential_6_simple_rnn_3_simple_rnn_cell_3_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :��������� : : : : : *%
_read_only_resource_inputs
	*6
body.R,
*sequential_6_simple_rnn_3_while_body_97256*6
cond.R,
*sequential_6_simple_rnn_3_while_cond_97255*8
output_shapes'
%: : : : :��������� : : : : : *
parallel_iterations 2!
sequential_6/simple_rnn_3/while�
Jsequential_6/simple_rnn_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2L
Jsequential_6/simple_rnn_3/TensorArrayV2Stack/TensorListStack/element_shape�
<sequential_6/simple_rnn_3/TensorArrayV2Stack/TensorListStackTensorListStack(sequential_6/simple_rnn_3/while:output:3Ssequential_6/simple_rnn_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������ *
element_dtype02>
<sequential_6/simple_rnn_3/TensorArrayV2Stack/TensorListStack�
/sequential_6/simple_rnn_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������21
/sequential_6/simple_rnn_3/strided_slice_3/stack�
1sequential_6/simple_rnn_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1sequential_6/simple_rnn_3/strided_slice_3/stack_1�
1sequential_6/simple_rnn_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential_6/simple_rnn_3/strided_slice_3/stack_2�
)sequential_6/simple_rnn_3/strided_slice_3StridedSliceEsequential_6/simple_rnn_3/TensorArrayV2Stack/TensorListStack:tensor:08sequential_6/simple_rnn_3/strided_slice_3/stack:output:0:sequential_6/simple_rnn_3/strided_slice_3/stack_1:output:0:sequential_6/simple_rnn_3/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2+
)sequential_6/simple_rnn_3/strided_slice_3�
*sequential_6/simple_rnn_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2,
*sequential_6/simple_rnn_3/transpose_1/perm�
%sequential_6/simple_rnn_3/transpose_1	TransposeEsequential_6/simple_rnn_3/TensorArrayV2Stack/TensorListStack:tensor:03sequential_6/simple_rnn_3/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2'
%sequential_6/simple_rnn_3/transpose_1�
sequential_6/simple_rnn_4/ShapeShape)sequential_6/simple_rnn_3/transpose_1:y:0*
T0*
_output_shapes
:2!
sequential_6/simple_rnn_4/Shape�
-sequential_6/simple_rnn_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_6/simple_rnn_4/strided_slice/stack�
/sequential_6/simple_rnn_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/sequential_6/simple_rnn_4/strided_slice/stack_1�
/sequential_6/simple_rnn_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/sequential_6/simple_rnn_4/strided_slice/stack_2�
'sequential_6/simple_rnn_4/strided_sliceStridedSlice(sequential_6/simple_rnn_4/Shape:output:06sequential_6/simple_rnn_4/strided_slice/stack:output:08sequential_6/simple_rnn_4/strided_slice/stack_1:output:08sequential_6/simple_rnn_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'sequential_6/simple_rnn_4/strided_slice�
%sequential_6/simple_rnn_4/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2'
%sequential_6/simple_rnn_4/zeros/mul/y�
#sequential_6/simple_rnn_4/zeros/mulMul0sequential_6/simple_rnn_4/strided_slice:output:0.sequential_6/simple_rnn_4/zeros/mul/y:output:0*
T0*
_output_shapes
: 2%
#sequential_6/simple_rnn_4/zeros/mul�
&sequential_6/simple_rnn_4/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2(
&sequential_6/simple_rnn_4/zeros/Less/y�
$sequential_6/simple_rnn_4/zeros/LessLess'sequential_6/simple_rnn_4/zeros/mul:z:0/sequential_6/simple_rnn_4/zeros/Less/y:output:0*
T0*
_output_shapes
: 2&
$sequential_6/simple_rnn_4/zeros/Less�
(sequential_6/simple_rnn_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_6/simple_rnn_4/zeros/packed/1�
&sequential_6/simple_rnn_4/zeros/packedPack0sequential_6/simple_rnn_4/strided_slice:output:01sequential_6/simple_rnn_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_6/simple_rnn_4/zeros/packed�
%sequential_6/simple_rnn_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%sequential_6/simple_rnn_4/zeros/Const�
sequential_6/simple_rnn_4/zerosFill/sequential_6/simple_rnn_4/zeros/packed:output:0.sequential_6/simple_rnn_4/zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2!
sequential_6/simple_rnn_4/zeros�
(sequential_6/simple_rnn_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(sequential_6/simple_rnn_4/transpose/perm�
#sequential_6/simple_rnn_4/transpose	Transpose)sequential_6/simple_rnn_3/transpose_1:y:01sequential_6/simple_rnn_4/transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2%
#sequential_6/simple_rnn_4/transpose�
!sequential_6/simple_rnn_4/Shape_1Shape'sequential_6/simple_rnn_4/transpose:y:0*
T0*
_output_shapes
:2#
!sequential_6/simple_rnn_4/Shape_1�
/sequential_6/simple_rnn_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/sequential_6/simple_rnn_4/strided_slice_1/stack�
1sequential_6/simple_rnn_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential_6/simple_rnn_4/strided_slice_1/stack_1�
1sequential_6/simple_rnn_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential_6/simple_rnn_4/strided_slice_1/stack_2�
)sequential_6/simple_rnn_4/strided_slice_1StridedSlice*sequential_6/simple_rnn_4/Shape_1:output:08sequential_6/simple_rnn_4/strided_slice_1/stack:output:0:sequential_6/simple_rnn_4/strided_slice_1/stack_1:output:0:sequential_6/simple_rnn_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)sequential_6/simple_rnn_4/strided_slice_1�
5sequential_6/simple_rnn_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������27
5sequential_6/simple_rnn_4/TensorArrayV2/element_shape�
'sequential_6/simple_rnn_4/TensorArrayV2TensorListReserve>sequential_6/simple_rnn_4/TensorArrayV2/element_shape:output:02sequential_6/simple_rnn_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'sequential_6/simple_rnn_4/TensorArrayV2�
Osequential_6/simple_rnn_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2Q
Osequential_6/simple_rnn_4/TensorArrayUnstack/TensorListFromTensor/element_shape�
Asequential_6/simple_rnn_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor'sequential_6/simple_rnn_4/transpose:y:0Xsequential_6/simple_rnn_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02C
Asequential_6/simple_rnn_4/TensorArrayUnstack/TensorListFromTensor�
/sequential_6/simple_rnn_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/sequential_6/simple_rnn_4/strided_slice_2/stack�
1sequential_6/simple_rnn_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential_6/simple_rnn_4/strided_slice_2/stack_1�
1sequential_6/simple_rnn_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential_6/simple_rnn_4/strided_slice_2/stack_2�
)sequential_6/simple_rnn_4/strided_slice_2StridedSlice'sequential_6/simple_rnn_4/transpose:y:08sequential_6/simple_rnn_4/strided_slice_2/stack:output:0:sequential_6/simple_rnn_4/strided_slice_2/stack_1:output:0:sequential_6/simple_rnn_4/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2+
)sequential_6/simple_rnn_4/strided_slice_2�
Asequential_6/simple_rnn_4/simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOpJsequential_6_simple_rnn_4_simple_rnn_cell_4_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02C
Asequential_6/simple_rnn_4/simple_rnn_cell_4/MatMul/ReadVariableOp�
2sequential_6/simple_rnn_4/simple_rnn_cell_4/MatMulMatMul2sequential_6/simple_rnn_4/strided_slice_2:output:0Isequential_6/simple_rnn_4/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 24
2sequential_6/simple_rnn_4/simple_rnn_cell_4/MatMul�
Bsequential_6/simple_rnn_4/simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOpKsequential_6_simple_rnn_4_simple_rnn_cell_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02D
Bsequential_6/simple_rnn_4/simple_rnn_cell_4/BiasAdd/ReadVariableOp�
3sequential_6/simple_rnn_4/simple_rnn_cell_4/BiasAddBiasAdd<sequential_6/simple_rnn_4/simple_rnn_cell_4/MatMul:product:0Jsequential_6/simple_rnn_4/simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 25
3sequential_6/simple_rnn_4/simple_rnn_cell_4/BiasAdd�
Csequential_6/simple_rnn_4/simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOpLsequential_6_simple_rnn_4_simple_rnn_cell_4_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype02E
Csequential_6/simple_rnn_4/simple_rnn_cell_4/MatMul_1/ReadVariableOp�
4sequential_6/simple_rnn_4/simple_rnn_cell_4/MatMul_1MatMul(sequential_6/simple_rnn_4/zeros:output:0Ksequential_6/simple_rnn_4/simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 26
4sequential_6/simple_rnn_4/simple_rnn_cell_4/MatMul_1�
/sequential_6/simple_rnn_4/simple_rnn_cell_4/addAddV2<sequential_6/simple_rnn_4/simple_rnn_cell_4/BiasAdd:output:0>sequential_6/simple_rnn_4/simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 21
/sequential_6/simple_rnn_4/simple_rnn_cell_4/add�
0sequential_6/simple_rnn_4/simple_rnn_cell_4/TanhTanh3sequential_6/simple_rnn_4/simple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:��������� 22
0sequential_6/simple_rnn_4/simple_rnn_cell_4/Tanh�
7sequential_6/simple_rnn_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    29
7sequential_6/simple_rnn_4/TensorArrayV2_1/element_shape�
)sequential_6/simple_rnn_4/TensorArrayV2_1TensorListReserve@sequential_6/simple_rnn_4/TensorArrayV2_1/element_shape:output:02sequential_6/simple_rnn_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02+
)sequential_6/simple_rnn_4/TensorArrayV2_1�
sequential_6/simple_rnn_4/timeConst*
_output_shapes
: *
dtype0*
value	B : 2 
sequential_6/simple_rnn_4/time�
2sequential_6/simple_rnn_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������24
2sequential_6/simple_rnn_4/while/maximum_iterations�
,sequential_6/simple_rnn_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_6/simple_rnn_4/while/loop_counter�
sequential_6/simple_rnn_4/whileWhile5sequential_6/simple_rnn_4/while/loop_counter:output:0;sequential_6/simple_rnn_4/while/maximum_iterations:output:0'sequential_6/simple_rnn_4/time:output:02sequential_6/simple_rnn_4/TensorArrayV2_1:handle:0(sequential_6/simple_rnn_4/zeros:output:02sequential_6/simple_rnn_4/strided_slice_1:output:0Qsequential_6/simple_rnn_4/TensorArrayUnstack/TensorListFromTensor:output_handle:0Jsequential_6_simple_rnn_4_simple_rnn_cell_4_matmul_readvariableop_resourceKsequential_6_simple_rnn_4_simple_rnn_cell_4_biasadd_readvariableop_resourceLsequential_6_simple_rnn_4_simple_rnn_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :��������� : : : : : *%
_read_only_resource_inputs
	*6
body.R,
*sequential_6_simple_rnn_4_while_body_97364*6
cond.R,
*sequential_6_simple_rnn_4_while_cond_97363*8
output_shapes'
%: : : : :��������� : : : : : *
parallel_iterations 2!
sequential_6/simple_rnn_4/while�
Jsequential_6/simple_rnn_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2L
Jsequential_6/simple_rnn_4/TensorArrayV2Stack/TensorListStack/element_shape�
<sequential_6/simple_rnn_4/TensorArrayV2Stack/TensorListStackTensorListStack(sequential_6/simple_rnn_4/while:output:3Ssequential_6/simple_rnn_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������ *
element_dtype02>
<sequential_6/simple_rnn_4/TensorArrayV2Stack/TensorListStack�
/sequential_6/simple_rnn_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������21
/sequential_6/simple_rnn_4/strided_slice_3/stack�
1sequential_6/simple_rnn_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1sequential_6/simple_rnn_4/strided_slice_3/stack_1�
1sequential_6/simple_rnn_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential_6/simple_rnn_4/strided_slice_3/stack_2�
)sequential_6/simple_rnn_4/strided_slice_3StridedSliceEsequential_6/simple_rnn_4/TensorArrayV2Stack/TensorListStack:tensor:08sequential_6/simple_rnn_4/strided_slice_3/stack:output:0:sequential_6/simple_rnn_4/strided_slice_3/stack_1:output:0:sequential_6/simple_rnn_4/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2+
)sequential_6/simple_rnn_4/strided_slice_3�
*sequential_6/simple_rnn_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2,
*sequential_6/simple_rnn_4/transpose_1/perm�
%sequential_6/simple_rnn_4/transpose_1	TransposeEsequential_6/simple_rnn_4/TensorArrayV2Stack/TensorListStack:tensor:03sequential_6/simple_rnn_4/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2'
%sequential_6/simple_rnn_4/transpose_1�
sequential_6/simple_rnn_5/ShapeShape)sequential_6/simple_rnn_4/transpose_1:y:0*
T0*
_output_shapes
:2!
sequential_6/simple_rnn_5/Shape�
-sequential_6/simple_rnn_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_6/simple_rnn_5/strided_slice/stack�
/sequential_6/simple_rnn_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/sequential_6/simple_rnn_5/strided_slice/stack_1�
/sequential_6/simple_rnn_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/sequential_6/simple_rnn_5/strided_slice/stack_2�
'sequential_6/simple_rnn_5/strided_sliceStridedSlice(sequential_6/simple_rnn_5/Shape:output:06sequential_6/simple_rnn_5/strided_slice/stack:output:08sequential_6/simple_rnn_5/strided_slice/stack_1:output:08sequential_6/simple_rnn_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'sequential_6/simple_rnn_5/strided_slice�
%sequential_6/simple_rnn_5/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2'
%sequential_6/simple_rnn_5/zeros/mul/y�
#sequential_6/simple_rnn_5/zeros/mulMul0sequential_6/simple_rnn_5/strided_slice:output:0.sequential_6/simple_rnn_5/zeros/mul/y:output:0*
T0*
_output_shapes
: 2%
#sequential_6/simple_rnn_5/zeros/mul�
&sequential_6/simple_rnn_5/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2(
&sequential_6/simple_rnn_5/zeros/Less/y�
$sequential_6/simple_rnn_5/zeros/LessLess'sequential_6/simple_rnn_5/zeros/mul:z:0/sequential_6/simple_rnn_5/zeros/Less/y:output:0*
T0*
_output_shapes
: 2&
$sequential_6/simple_rnn_5/zeros/Less�
(sequential_6/simple_rnn_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_6/simple_rnn_5/zeros/packed/1�
&sequential_6/simple_rnn_5/zeros/packedPack0sequential_6/simple_rnn_5/strided_slice:output:01sequential_6/simple_rnn_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_6/simple_rnn_5/zeros/packed�
%sequential_6/simple_rnn_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%sequential_6/simple_rnn_5/zeros/Const�
sequential_6/simple_rnn_5/zerosFill/sequential_6/simple_rnn_5/zeros/packed:output:0.sequential_6/simple_rnn_5/zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2!
sequential_6/simple_rnn_5/zeros�
(sequential_6/simple_rnn_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(sequential_6/simple_rnn_5/transpose/perm�
#sequential_6/simple_rnn_5/transpose	Transpose)sequential_6/simple_rnn_4/transpose_1:y:01sequential_6/simple_rnn_5/transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2%
#sequential_6/simple_rnn_5/transpose�
!sequential_6/simple_rnn_5/Shape_1Shape'sequential_6/simple_rnn_5/transpose:y:0*
T0*
_output_shapes
:2#
!sequential_6/simple_rnn_5/Shape_1�
/sequential_6/simple_rnn_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/sequential_6/simple_rnn_5/strided_slice_1/stack�
1sequential_6/simple_rnn_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential_6/simple_rnn_5/strided_slice_1/stack_1�
1sequential_6/simple_rnn_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential_6/simple_rnn_5/strided_slice_1/stack_2�
)sequential_6/simple_rnn_5/strided_slice_1StridedSlice*sequential_6/simple_rnn_5/Shape_1:output:08sequential_6/simple_rnn_5/strided_slice_1/stack:output:0:sequential_6/simple_rnn_5/strided_slice_1/stack_1:output:0:sequential_6/simple_rnn_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)sequential_6/simple_rnn_5/strided_slice_1�
5sequential_6/simple_rnn_5/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������27
5sequential_6/simple_rnn_5/TensorArrayV2/element_shape�
'sequential_6/simple_rnn_5/TensorArrayV2TensorListReserve>sequential_6/simple_rnn_5/TensorArrayV2/element_shape:output:02sequential_6/simple_rnn_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'sequential_6/simple_rnn_5/TensorArrayV2�
Osequential_6/simple_rnn_5/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2Q
Osequential_6/simple_rnn_5/TensorArrayUnstack/TensorListFromTensor/element_shape�
Asequential_6/simple_rnn_5/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor'sequential_6/simple_rnn_5/transpose:y:0Xsequential_6/simple_rnn_5/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02C
Asequential_6/simple_rnn_5/TensorArrayUnstack/TensorListFromTensor�
/sequential_6/simple_rnn_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/sequential_6/simple_rnn_5/strided_slice_2/stack�
1sequential_6/simple_rnn_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential_6/simple_rnn_5/strided_slice_2/stack_1�
1sequential_6/simple_rnn_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential_6/simple_rnn_5/strided_slice_2/stack_2�
)sequential_6/simple_rnn_5/strided_slice_2StridedSlice'sequential_6/simple_rnn_5/transpose:y:08sequential_6/simple_rnn_5/strided_slice_2/stack:output:0:sequential_6/simple_rnn_5/strided_slice_2/stack_1:output:0:sequential_6/simple_rnn_5/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2+
)sequential_6/simple_rnn_5/strided_slice_2�
Asequential_6/simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOpJsequential_6_simple_rnn_5_simple_rnn_cell_5_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02C
Asequential_6/simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOp�
2sequential_6/simple_rnn_5/simple_rnn_cell_5/MatMulMatMul2sequential_6/simple_rnn_5/strided_slice_2:output:0Isequential_6/simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 24
2sequential_6/simple_rnn_5/simple_rnn_cell_5/MatMul�
Bsequential_6/simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOpKsequential_6_simple_rnn_5_simple_rnn_cell_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02D
Bsequential_6/simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOp�
3sequential_6/simple_rnn_5/simple_rnn_cell_5/BiasAddBiasAdd<sequential_6/simple_rnn_5/simple_rnn_cell_5/MatMul:product:0Jsequential_6/simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 25
3sequential_6/simple_rnn_5/simple_rnn_cell_5/BiasAdd�
Csequential_6/simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOpLsequential_6_simple_rnn_5_simple_rnn_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype02E
Csequential_6/simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp�
4sequential_6/simple_rnn_5/simple_rnn_cell_5/MatMul_1MatMul(sequential_6/simple_rnn_5/zeros:output:0Ksequential_6/simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 26
4sequential_6/simple_rnn_5/simple_rnn_cell_5/MatMul_1�
/sequential_6/simple_rnn_5/simple_rnn_cell_5/addAddV2<sequential_6/simple_rnn_5/simple_rnn_cell_5/BiasAdd:output:0>sequential_6/simple_rnn_5/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 21
/sequential_6/simple_rnn_5/simple_rnn_cell_5/add�
0sequential_6/simple_rnn_5/simple_rnn_cell_5/TanhTanh3sequential_6/simple_rnn_5/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:��������� 22
0sequential_6/simple_rnn_5/simple_rnn_cell_5/Tanh�
7sequential_6/simple_rnn_5/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    29
7sequential_6/simple_rnn_5/TensorArrayV2_1/element_shape�
)sequential_6/simple_rnn_5/TensorArrayV2_1TensorListReserve@sequential_6/simple_rnn_5/TensorArrayV2_1/element_shape:output:02sequential_6/simple_rnn_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02+
)sequential_6/simple_rnn_5/TensorArrayV2_1�
sequential_6/simple_rnn_5/timeConst*
_output_shapes
: *
dtype0*
value	B : 2 
sequential_6/simple_rnn_5/time�
2sequential_6/simple_rnn_5/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������24
2sequential_6/simple_rnn_5/while/maximum_iterations�
,sequential_6/simple_rnn_5/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_6/simple_rnn_5/while/loop_counter�
sequential_6/simple_rnn_5/whileWhile5sequential_6/simple_rnn_5/while/loop_counter:output:0;sequential_6/simple_rnn_5/while/maximum_iterations:output:0'sequential_6/simple_rnn_5/time:output:02sequential_6/simple_rnn_5/TensorArrayV2_1:handle:0(sequential_6/simple_rnn_5/zeros:output:02sequential_6/simple_rnn_5/strided_slice_1:output:0Qsequential_6/simple_rnn_5/TensorArrayUnstack/TensorListFromTensor:output_handle:0Jsequential_6_simple_rnn_5_simple_rnn_cell_5_matmul_readvariableop_resourceKsequential_6_simple_rnn_5_simple_rnn_cell_5_biasadd_readvariableop_resourceLsequential_6_simple_rnn_5_simple_rnn_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :��������� : : : : : *%
_read_only_resource_inputs
	*6
body.R,
*sequential_6_simple_rnn_5_while_body_97472*6
cond.R,
*sequential_6_simple_rnn_5_while_cond_97471*8
output_shapes'
%: : : : :��������� : : : : : *
parallel_iterations 2!
sequential_6/simple_rnn_5/while�
Jsequential_6/simple_rnn_5/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2L
Jsequential_6/simple_rnn_5/TensorArrayV2Stack/TensorListStack/element_shape�
<sequential_6/simple_rnn_5/TensorArrayV2Stack/TensorListStackTensorListStack(sequential_6/simple_rnn_5/while:output:3Ssequential_6/simple_rnn_5/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������ *
element_dtype02>
<sequential_6/simple_rnn_5/TensorArrayV2Stack/TensorListStack�
/sequential_6/simple_rnn_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������21
/sequential_6/simple_rnn_5/strided_slice_3/stack�
1sequential_6/simple_rnn_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1sequential_6/simple_rnn_5/strided_slice_3/stack_1�
1sequential_6/simple_rnn_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential_6/simple_rnn_5/strided_slice_3/stack_2�
)sequential_6/simple_rnn_5/strided_slice_3StridedSliceEsequential_6/simple_rnn_5/TensorArrayV2Stack/TensorListStack:tensor:08sequential_6/simple_rnn_5/strided_slice_3/stack:output:0:sequential_6/simple_rnn_5/strided_slice_3/stack_1:output:0:sequential_6/simple_rnn_5/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2+
)sequential_6/simple_rnn_5/strided_slice_3�
*sequential_6/simple_rnn_5/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2,
*sequential_6/simple_rnn_5/transpose_1/perm�
%sequential_6/simple_rnn_5/transpose_1	TransposeEsequential_6/simple_rnn_5/TensorArrayV2Stack/TensorListStack:tensor:03sequential_6/simple_rnn_5/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2'
%sequential_6/simple_rnn_5/transpose_1�
+sequential_6/dense_22/MatMul/ReadVariableOpReadVariableOp4sequential_6_dense_22_matmul_readvariableop_resource*
_output_shapes

: *
dtype02-
+sequential_6/dense_22/MatMul/ReadVariableOp�
sequential_6/dense_22/MatMulMatMul2sequential_6/simple_rnn_5/strided_slice_3:output:03sequential_6/dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sequential_6/dense_22/MatMul�
,sequential_6/dense_22/BiasAdd/ReadVariableOpReadVariableOp5sequential_6_dense_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_6/dense_22/BiasAdd/ReadVariableOp�
sequential_6/dense_22/BiasAddBiasAdd&sequential_6/dense_22/MatMul:product:04sequential_6/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sequential_6/dense_22/BiasAdd�
sequential_6/dense_22/ReluRelu&sequential_6/dense_22/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
sequential_6/dense_22/Relu�
+sequential_6/dense_23/MatMul/ReadVariableOpReadVariableOp4sequential_6_dense_23_matmul_readvariableop_resource*
_output_shapes

:*
dtype02-
+sequential_6/dense_23/MatMul/ReadVariableOp�
sequential_6/dense_23/MatMulMatMul(sequential_6/dense_22/Relu:activations:03sequential_6/dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sequential_6/dense_23/MatMul�
,sequential_6/dense_23/BiasAdd/ReadVariableOpReadVariableOp5sequential_6_dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_6/dense_23/BiasAdd/ReadVariableOp�
sequential_6/dense_23/BiasAddBiasAdd&sequential_6/dense_23/MatMul:product:04sequential_6/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sequential_6/dense_23/BiasAdd�
IdentityIdentity&sequential_6/dense_23/BiasAdd:output:0-^sequential_6/dense_22/BiasAdd/ReadVariableOp,^sequential_6/dense_22/MatMul/ReadVariableOp-^sequential_6/dense_23/BiasAdd/ReadVariableOp,^sequential_6/dense_23/MatMul/ReadVariableOpC^sequential_6/simple_rnn_3/simple_rnn_cell_3/BiasAdd/ReadVariableOpB^sequential_6/simple_rnn_3/simple_rnn_cell_3/MatMul/ReadVariableOpD^sequential_6/simple_rnn_3/simple_rnn_cell_3/MatMul_1/ReadVariableOp ^sequential_6/simple_rnn_3/whileC^sequential_6/simple_rnn_4/simple_rnn_cell_4/BiasAdd/ReadVariableOpB^sequential_6/simple_rnn_4/simple_rnn_cell_4/MatMul/ReadVariableOpD^sequential_6/simple_rnn_4/simple_rnn_cell_4/MatMul_1/ReadVariableOp ^sequential_6/simple_rnn_4/whileC^sequential_6/simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOpB^sequential_6/simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOpD^sequential_6/simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp ^sequential_6/simple_rnn_5/while*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*c
_input_shapesR
P:������������������:::::::::::::2\
,sequential_6/dense_22/BiasAdd/ReadVariableOp,sequential_6/dense_22/BiasAdd/ReadVariableOp2Z
+sequential_6/dense_22/MatMul/ReadVariableOp+sequential_6/dense_22/MatMul/ReadVariableOp2\
,sequential_6/dense_23/BiasAdd/ReadVariableOp,sequential_6/dense_23/BiasAdd/ReadVariableOp2Z
+sequential_6/dense_23/MatMul/ReadVariableOp+sequential_6/dense_23/MatMul/ReadVariableOp2�
Bsequential_6/simple_rnn_3/simple_rnn_cell_3/BiasAdd/ReadVariableOpBsequential_6/simple_rnn_3/simple_rnn_cell_3/BiasAdd/ReadVariableOp2�
Asequential_6/simple_rnn_3/simple_rnn_cell_3/MatMul/ReadVariableOpAsequential_6/simple_rnn_3/simple_rnn_cell_3/MatMul/ReadVariableOp2�
Csequential_6/simple_rnn_3/simple_rnn_cell_3/MatMul_1/ReadVariableOpCsequential_6/simple_rnn_3/simple_rnn_cell_3/MatMul_1/ReadVariableOp2B
sequential_6/simple_rnn_3/whilesequential_6/simple_rnn_3/while2�
Bsequential_6/simple_rnn_4/simple_rnn_cell_4/BiasAdd/ReadVariableOpBsequential_6/simple_rnn_4/simple_rnn_cell_4/BiasAdd/ReadVariableOp2�
Asequential_6/simple_rnn_4/simple_rnn_cell_4/MatMul/ReadVariableOpAsequential_6/simple_rnn_4/simple_rnn_cell_4/MatMul/ReadVariableOp2�
Csequential_6/simple_rnn_4/simple_rnn_cell_4/MatMul_1/ReadVariableOpCsequential_6/simple_rnn_4/simple_rnn_cell_4/MatMul_1/ReadVariableOp2B
sequential_6/simple_rnn_4/whilesequential_6/simple_rnn_4/while2�
Bsequential_6/simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOpBsequential_6/simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOp2�
Asequential_6/simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOpAsequential_6/simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOp2�
Csequential_6/simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOpCsequential_6/simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp2B
sequential_6/simple_rnn_5/whilesequential_6/simple_rnn_5/while:` \
0
_output_shapes
:������������������
(
_user_specified_namelambda_1_input
�
�
while_cond_101314
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_101314___redundant_placeholder04
0while_while_cond_101314___redundant_placeholder14
0while_while_cond_101314___redundant_placeholder24
0while_while_cond_101314___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�
�
-__inference_simple_rnn_3_layer_call_fn_101392

inputs
unknown
	unknown_0
	unknown_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_992272
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :������������������ 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�Q
�
*sequential_6_simple_rnn_4_while_body_97364P
Lsequential_6_simple_rnn_4_while_sequential_6_simple_rnn_4_while_loop_counterV
Rsequential_6_simple_rnn_4_while_sequential_6_simple_rnn_4_while_maximum_iterations/
+sequential_6_simple_rnn_4_while_placeholder1
-sequential_6_simple_rnn_4_while_placeholder_11
-sequential_6_simple_rnn_4_while_placeholder_2O
Ksequential_6_simple_rnn_4_while_sequential_6_simple_rnn_4_strided_slice_1_0�
�sequential_6_simple_rnn_4_while_tensorarrayv2read_tensorlistgetitem_sequential_6_simple_rnn_4_tensorarrayunstack_tensorlistfromtensor_0V
Rsequential_6_simple_rnn_4_while_simple_rnn_cell_4_matmul_readvariableop_resource_0W
Ssequential_6_simple_rnn_4_while_simple_rnn_cell_4_biasadd_readvariableop_resource_0X
Tsequential_6_simple_rnn_4_while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0,
(sequential_6_simple_rnn_4_while_identity.
*sequential_6_simple_rnn_4_while_identity_1.
*sequential_6_simple_rnn_4_while_identity_2.
*sequential_6_simple_rnn_4_while_identity_3.
*sequential_6_simple_rnn_4_while_identity_4M
Isequential_6_simple_rnn_4_while_sequential_6_simple_rnn_4_strided_slice_1�
�sequential_6_simple_rnn_4_while_tensorarrayv2read_tensorlistgetitem_sequential_6_simple_rnn_4_tensorarrayunstack_tensorlistfromtensorT
Psequential_6_simple_rnn_4_while_simple_rnn_cell_4_matmul_readvariableop_resourceU
Qsequential_6_simple_rnn_4_while_simple_rnn_cell_4_biasadd_readvariableop_resourceV
Rsequential_6_simple_rnn_4_while_simple_rnn_cell_4_matmul_1_readvariableop_resource��Hsequential_6/simple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp�Gsequential_6/simple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOp�Isequential_6/simple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp�
Qsequential_6/simple_rnn_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2S
Qsequential_6/simple_rnn_4/while/TensorArrayV2Read/TensorListGetItem/element_shape�
Csequential_6/simple_rnn_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem�sequential_6_simple_rnn_4_while_tensorarrayv2read_tensorlistgetitem_sequential_6_simple_rnn_4_tensorarrayunstack_tensorlistfromtensor_0+sequential_6_simple_rnn_4_while_placeholderZsequential_6/simple_rnn_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:��������� *
element_dtype02E
Csequential_6/simple_rnn_4/while/TensorArrayV2Read/TensorListGetItem�
Gsequential_6/simple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOpRsequential_6_simple_rnn_4_while_simple_rnn_cell_4_matmul_readvariableop_resource_0*
_output_shapes

:  *
dtype02I
Gsequential_6/simple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOp�
8sequential_6/simple_rnn_4/while/simple_rnn_cell_4/MatMulMatMulJsequential_6/simple_rnn_4/while/TensorArrayV2Read/TensorListGetItem:item:0Osequential_6/simple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2:
8sequential_6/simple_rnn_4/while/simple_rnn_cell_4/MatMul�
Hsequential_6/simple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOpSsequential_6_simple_rnn_4_while_simple_rnn_cell_4_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype02J
Hsequential_6/simple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp�
9sequential_6/simple_rnn_4/while/simple_rnn_cell_4/BiasAddBiasAddBsequential_6/simple_rnn_4/while/simple_rnn_cell_4/MatMul:product:0Psequential_6/simple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2;
9sequential_6/simple_rnn_4/while/simple_rnn_cell_4/BiasAdd�
Isequential_6/simple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOpTsequential_6_simple_rnn_4_while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype02K
Isequential_6/simple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp�
:sequential_6/simple_rnn_4/while/simple_rnn_cell_4/MatMul_1MatMul-sequential_6_simple_rnn_4_while_placeholder_2Qsequential_6/simple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2<
:sequential_6/simple_rnn_4/while/simple_rnn_cell_4/MatMul_1�
5sequential_6/simple_rnn_4/while/simple_rnn_cell_4/addAddV2Bsequential_6/simple_rnn_4/while/simple_rnn_cell_4/BiasAdd:output:0Dsequential_6/simple_rnn_4/while/simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 27
5sequential_6/simple_rnn_4/while/simple_rnn_cell_4/add�
6sequential_6/simple_rnn_4/while/simple_rnn_cell_4/TanhTanh9sequential_6/simple_rnn_4/while/simple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:��������� 28
6sequential_6/simple_rnn_4/while/simple_rnn_cell_4/Tanh�
Dsequential_6/simple_rnn_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem-sequential_6_simple_rnn_4_while_placeholder_1+sequential_6_simple_rnn_4_while_placeholder:sequential_6/simple_rnn_4/while/simple_rnn_cell_4/Tanh:y:0*
_output_shapes
: *
element_dtype02F
Dsequential_6/simple_rnn_4/while/TensorArrayV2Write/TensorListSetItem�
%sequential_6/simple_rnn_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_6/simple_rnn_4/while/add/y�
#sequential_6/simple_rnn_4/while/addAddV2+sequential_6_simple_rnn_4_while_placeholder.sequential_6/simple_rnn_4/while/add/y:output:0*
T0*
_output_shapes
: 2%
#sequential_6/simple_rnn_4/while/add�
'sequential_6/simple_rnn_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_6/simple_rnn_4/while/add_1/y�
%sequential_6/simple_rnn_4/while/add_1AddV2Lsequential_6_simple_rnn_4_while_sequential_6_simple_rnn_4_while_loop_counter0sequential_6/simple_rnn_4/while/add_1/y:output:0*
T0*
_output_shapes
: 2'
%sequential_6/simple_rnn_4/while/add_1�
(sequential_6/simple_rnn_4/while/IdentityIdentity)sequential_6/simple_rnn_4/while/add_1:z:0I^sequential_6/simple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOpH^sequential_6/simple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOpJ^sequential_6/simple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2*
(sequential_6/simple_rnn_4/while/Identity�
*sequential_6/simple_rnn_4/while/Identity_1IdentityRsequential_6_simple_rnn_4_while_sequential_6_simple_rnn_4_while_maximum_iterationsI^sequential_6/simple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOpH^sequential_6/simple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOpJ^sequential_6/simple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2,
*sequential_6/simple_rnn_4/while/Identity_1�
*sequential_6/simple_rnn_4/while/Identity_2Identity'sequential_6/simple_rnn_4/while/add:z:0I^sequential_6/simple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOpH^sequential_6/simple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOpJ^sequential_6/simple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2,
*sequential_6/simple_rnn_4/while/Identity_2�
*sequential_6/simple_rnn_4/while/Identity_3IdentityTsequential_6/simple_rnn_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0I^sequential_6/simple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOpH^sequential_6/simple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOpJ^sequential_6/simple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2,
*sequential_6/simple_rnn_4/while/Identity_3�
*sequential_6/simple_rnn_4/while/Identity_4Identity:sequential_6/simple_rnn_4/while/simple_rnn_cell_4/Tanh:y:0I^sequential_6/simple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOpH^sequential_6/simple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOpJ^sequential_6/simple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2,
*sequential_6/simple_rnn_4/while/Identity_4"]
(sequential_6_simple_rnn_4_while_identity1sequential_6/simple_rnn_4/while/Identity:output:0"a
*sequential_6_simple_rnn_4_while_identity_13sequential_6/simple_rnn_4/while/Identity_1:output:0"a
*sequential_6_simple_rnn_4_while_identity_23sequential_6/simple_rnn_4/while/Identity_2:output:0"a
*sequential_6_simple_rnn_4_while_identity_33sequential_6/simple_rnn_4/while/Identity_3:output:0"a
*sequential_6_simple_rnn_4_while_identity_43sequential_6/simple_rnn_4/while/Identity_4:output:0"�
Isequential_6_simple_rnn_4_while_sequential_6_simple_rnn_4_strided_slice_1Ksequential_6_simple_rnn_4_while_sequential_6_simple_rnn_4_strided_slice_1_0"�
Qsequential_6_simple_rnn_4_while_simple_rnn_cell_4_biasadd_readvariableop_resourceSsequential_6_simple_rnn_4_while_simple_rnn_cell_4_biasadd_readvariableop_resource_0"�
Rsequential_6_simple_rnn_4_while_simple_rnn_cell_4_matmul_1_readvariableop_resourceTsequential_6_simple_rnn_4_while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0"�
Psequential_6_simple_rnn_4_while_simple_rnn_cell_4_matmul_readvariableop_resourceRsequential_6_simple_rnn_4_while_simple_rnn_cell_4_matmul_readvariableop_resource_0"�
�sequential_6_simple_rnn_4_while_tensorarrayv2read_tensorlistgetitem_sequential_6_simple_rnn_4_tensorarrayunstack_tensorlistfromtensor�sequential_6_simple_rnn_4_while_tensorarrayv2read_tensorlistgetitem_sequential_6_simple_rnn_4_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :��������� : : :::2�
Hsequential_6/simple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOpHsequential_6/simple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp2�
Gsequential_6/simple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOpGsequential_6/simple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOp2�
Isequential_6/simple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpIsequential_6/simple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�H
�
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_101515

inputs4
0simple_rnn_cell_4_matmul_readvariableop_resource5
1simple_rnn_cell_4_biasadd_readvariableop_resource6
2simple_rnn_cell_4_matmul_1_readvariableop_resource
identity��(simple_rnn_cell_4/BiasAdd/ReadVariableOp�'simple_rnn_cell_4/MatMul/ReadVariableOp�)simple_rnn_cell_4/MatMul_1/ReadVariableOp�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
strided_slice_2�
'simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_4_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02)
'simple_rnn_cell_4/MatMul/ReadVariableOp�
simple_rnn_cell_4/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_4/MatMul�
(simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(simple_rnn_cell_4/BiasAdd/ReadVariableOp�
simple_rnn_cell_4/BiasAddBiasAdd"simple_rnn_cell_4/MatMul:product:00simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_4/BiasAdd�
)simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_4_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype02+
)simple_rnn_cell_4/MatMul_1/ReadVariableOp�
simple_rnn_cell_4/MatMul_1MatMulzeros:output:01simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_4/MatMul_1�
simple_rnn_cell_4/addAddV2"simple_rnn_cell_4/BiasAdd:output:0$simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_4/add�
simple_rnn_cell_4/TanhTanhsimple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_4/Tanh�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_4_matmul_readvariableop_resource1simple_rnn_cell_4_biasadd_readvariableop_resource2simple_rnn_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :��������� : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_101449*
condR
while_cond_101448*8
output_shapes'
%: : : : :��������� : : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
transpose_1�
IdentityIdentitytranspose_1:y:0)^simple_rnn_cell_4/BiasAdd/ReadVariableOp(^simple_rnn_cell_4/MatMul/ReadVariableOp*^simple_rnn_cell_4/MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :������������������ 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������ :::2T
(simple_rnn_cell_4/BiasAdd/ReadVariableOp(simple_rnn_cell_4/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_4/MatMul/ReadVariableOp'simple_rnn_cell_4/MatMul/ReadVariableOp2V
)simple_rnn_cell_4/MatMul_1/ReadVariableOp)simple_rnn_cell_4/MatMul_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�H
�
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_102365

inputs4
0simple_rnn_cell_5_matmul_readvariableop_resource5
1simple_rnn_cell_5_biasadd_readvariableop_resource6
2simple_rnn_cell_5_matmul_1_readvariableop_resource
identity��(simple_rnn_cell_5/BiasAdd/ReadVariableOp�'simple_rnn_cell_5/MatMul/ReadVariableOp�)simple_rnn_cell_5/MatMul_1/ReadVariableOp�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
strided_slice_2�
'simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_5_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02)
'simple_rnn_cell_5/MatMul/ReadVariableOp�
simple_rnn_cell_5/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_5/MatMul�
(simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(simple_rnn_cell_5/BiasAdd/ReadVariableOp�
simple_rnn_cell_5/BiasAddBiasAdd"simple_rnn_cell_5/MatMul:product:00simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_5/BiasAdd�
)simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype02+
)simple_rnn_cell_5/MatMul_1/ReadVariableOp�
simple_rnn_cell_5/MatMul_1MatMulzeros:output:01simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_5/MatMul_1�
simple_rnn_cell_5/addAddV2"simple_rnn_cell_5/BiasAdd:output:0$simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_5/add�
simple_rnn_cell_5/TanhTanhsimple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_5/Tanh�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_5_matmul_readvariableop_resource1simple_rnn_cell_5_biasadd_readvariableop_resource2simple_rnn_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :��������� : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_102299*
condR
while_cond_102298*8
output_shapes'
%: : : : :��������� : : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
transpose_1�
IdentityIdentitystrided_slice_3:output:0)^simple_rnn_cell_5/BiasAdd/ReadVariableOp(^simple_rnn_cell_5/MatMul/ReadVariableOp*^simple_rnn_cell_5/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������ :::2T
(simple_rnn_cell_5/BiasAdd/ReadVariableOp(simple_rnn_cell_5/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_5/MatMul/ReadVariableOp'simple_rnn_cell_5/MatMul/ReadVariableOp2V
)simple_rnn_cell_5/MatMul_1/ReadVariableOp)simple_rnn_cell_5/MatMul_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�
�
while_cond_101940
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_101940___redundant_placeholder04
0while_while_cond_101940___redundant_placeholder14
0while_while_cond_101940___redundant_placeholder24
0while_while_cond_101940___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�!
�
H__inference_sequential_6_layer_call_and_return_conditional_losses_100071

inputs
simple_rnn_3_100039
simple_rnn_3_100041
simple_rnn_3_100043
simple_rnn_4_100046
simple_rnn_4_100048
simple_rnn_4_100050
simple_rnn_5_100053
simple_rnn_5_100055
simple_rnn_5_100057
dense_22_100060
dense_22_100062
dense_23_100065
dense_23_100067
identity�� dense_22/StatefulPartitionedCall� dense_23/StatefulPartitionedCall�$simple_rnn_3/StatefulPartitionedCall�$simple_rnn_4/StatefulPartitionedCall�$simple_rnn_5/StatefulPartitionedCall�
lambda_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_lambda_1_layer_call_and_return_conditional_losses_991032
lambda_1/PartitionedCall�
$simple_rnn_3/StatefulPartitionedCallStatefulPartitionedCall!lambda_1/PartitionedCall:output:0simple_rnn_3_100039simple_rnn_3_100041simple_rnn_3_100043*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_993392&
$simple_rnn_3/StatefulPartitionedCall�
$simple_rnn_4/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_3/StatefulPartitionedCall:output:0simple_rnn_4_100046simple_rnn_4_100048simple_rnn_4_100050*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_simple_rnn_4_layer_call_and_return_conditional_losses_995922&
$simple_rnn_4/StatefulPartitionedCall�
$simple_rnn_5/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_4/StatefulPartitionedCall:output:0simple_rnn_5_100053simple_rnn_5_100055simple_rnn_5_100057*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_simple_rnn_5_layer_call_and_return_conditional_losses_998452&
$simple_rnn_5/StatefulPartitionedCall�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_5/StatefulPartitionedCall:output:0dense_22_100060dense_22_100062*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_22_layer_call_and_return_conditional_losses_998862"
 dense_22/StatefulPartitionedCall�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_100065dense_23_100067*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_23_layer_call_and_return_conditional_losses_999122"
 dense_23/StatefulPartitionedCall�
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall%^simple_rnn_3/StatefulPartitionedCall%^simple_rnn_4/StatefulPartitionedCall%^simple_rnn_5/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*c
_input_shapesR
P:������������������:::::::::::::2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2L
$simple_rnn_3/StatefulPartitionedCall$simple_rnn_3/StatefulPartitionedCall2L
$simple_rnn_4/StatefulPartitionedCall$simple_rnn_4/StatefulPartitionedCall2L
$simple_rnn_5/StatefulPartitionedCall$simple_rnn_5/StatefulPartitionedCall:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�H
�
H__inference_simple_rnn_3_layer_call_and_return_conditional_losses_101269

inputs4
0simple_rnn_cell_3_matmul_readvariableop_resource5
1simple_rnn_cell_3_biasadd_readvariableop_resource6
2simple_rnn_cell_3_matmul_1_readvariableop_resource
identity��(simple_rnn_cell_3/BiasAdd/ReadVariableOp�'simple_rnn_cell_3/MatMul/ReadVariableOp�)simple_rnn_cell_3/MatMul_1/ReadVariableOp�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
'simple_rnn_cell_3/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02)
'simple_rnn_cell_3/MatMul/ReadVariableOp�
simple_rnn_cell_3/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_3/MatMul�
(simple_rnn_cell_3/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(simple_rnn_cell_3/BiasAdd/ReadVariableOp�
simple_rnn_cell_3/BiasAddBiasAdd"simple_rnn_cell_3/MatMul:product:00simple_rnn_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_3/BiasAdd�
)simple_rnn_cell_3/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_3_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype02+
)simple_rnn_cell_3/MatMul_1/ReadVariableOp�
simple_rnn_cell_3/MatMul_1MatMulzeros:output:01simple_rnn_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_3/MatMul_1�
simple_rnn_cell_3/addAddV2"simple_rnn_cell_3/BiasAdd:output:0$simple_rnn_cell_3/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_3/add�
simple_rnn_cell_3/TanhTanhsimple_rnn_cell_3/add:z:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_3/Tanh�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_3_matmul_readvariableop_resource1simple_rnn_cell_3_biasadd_readvariableop_resource2simple_rnn_cell_3_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :��������� : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_101203*
condR
while_cond_101202*8
output_shapes'
%: : : : :��������� : : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
transpose_1�
IdentityIdentitytranspose_1:y:0)^simple_rnn_cell_3/BiasAdd/ReadVariableOp(^simple_rnn_cell_3/MatMul/ReadVariableOp*^simple_rnn_cell_3/MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :������������������ 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::2T
(simple_rnn_cell_3/BiasAdd/ReadVariableOp(simple_rnn_cell_3/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_3/MatMul/ReadVariableOp'simple_rnn_cell_3/MatMul/ReadVariableOp2V
)simple_rnn_cell_3/MatMul_1/ReadVariableOp)simple_rnn_cell_3/MatMul_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�3
�
while_body_101315
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_3_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_3_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_3_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_3_matmul_readvariableop_resource;
7while_simple_rnn_cell_3_biasadd_readvariableop_resource<
8while_simple_rnn_cell_3_matmul_1_readvariableop_resource��.while/simple_rnn_cell_3/BiasAdd/ReadVariableOp�-while/simple_rnn_cell_3/MatMul/ReadVariableOp�/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
-while/simple_rnn_cell_3/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_3_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype02/
-while/simple_rnn_cell_3/MatMul/ReadVariableOp�
while/simple_rnn_cell_3/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2 
while/simple_rnn_cell_3/MatMul�
.while/simple_rnn_cell_3/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_3_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype020
.while/simple_rnn_cell_3/BiasAdd/ReadVariableOp�
while/simple_rnn_cell_3/BiasAddBiasAdd(while/simple_rnn_cell_3/MatMul:product:06while/simple_rnn_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2!
while/simple_rnn_cell_3/BiasAdd�
/while/simple_rnn_cell_3/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_3_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype021
/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp�
 while/simple_rnn_cell_3/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2"
 while/simple_rnn_cell_3/MatMul_1�
while/simple_rnn_cell_3/addAddV2(while/simple_rnn_cell_3/BiasAdd:output:0*while/simple_rnn_cell_3/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2
while/simple_rnn_cell_3/add�
while/simple_rnn_cell_3/TanhTanhwhile/simple_rnn_cell_3/add:z:0*
T0*'
_output_shapes
:��������� 2
while/simple_rnn_cell_3/Tanh�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_3/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_3/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_3/MatMul/ReadVariableOp0^while/simple_rnn_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_3/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_3/MatMul/ReadVariableOp0^while/simple_rnn_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_3/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_3/MatMul/ReadVariableOp0^while/simple_rnn_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_3/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_3/MatMul/ReadVariableOp0^while/simple_rnn_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity while/simple_rnn_cell_3/Tanh:y:0/^while/simple_rnn_cell_3/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_3/MatMul/ReadVariableOp0^while/simple_rnn_cell_3/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_3_biasadd_readvariableop_resource9while_simple_rnn_cell_3_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_3_matmul_1_readvariableop_resource:while_simple_rnn_cell_3_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_3_matmul_readvariableop_resource8while_simple_rnn_cell_3_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :��������� : : :::2`
.while/simple_rnn_cell_3/BiasAdd/ReadVariableOp.while/simple_rnn_cell_3/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_3/MatMul/ReadVariableOp-while/simple_rnn_cell_3/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp/while/simple_rnn_cell_3/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�<
�
G__inference_simple_rnn_5_layer_call_and_return_conditional_losses_98961

inputs
simple_rnn_cell_5_98886
simple_rnn_cell_5_98888
simple_rnn_cell_5_98890
identity��)simple_rnn_cell_5/StatefulPartitionedCall�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
strided_slice_2�
)simple_rnn_cell_5/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_5_98886simple_rnn_cell_5_98888simple_rnn_cell_5_98890*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:��������� :��������� *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_986242+
)simple_rnn_cell_5/StatefulPartitionedCall�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_5_98886simple_rnn_cell_5_98888simple_rnn_cell_5_98890*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :��������� : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_98898*
condR
while_cond_98897*8
output_shapes'
%: : : : :��������� : : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
transpose_1�
IdentityIdentitystrided_slice_3:output:0*^simple_rnn_cell_5/StatefulPartitionedCall^while*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������ :::2V
)simple_rnn_cell_5/StatefulPartitionedCall)simple_rnn_cell_5/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�H
�
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_101627

inputs4
0simple_rnn_cell_4_matmul_readvariableop_resource5
1simple_rnn_cell_4_biasadd_readvariableop_resource6
2simple_rnn_cell_4_matmul_1_readvariableop_resource
identity��(simple_rnn_cell_4/BiasAdd/ReadVariableOp�'simple_rnn_cell_4/MatMul/ReadVariableOp�)simple_rnn_cell_4/MatMul_1/ReadVariableOp�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
strided_slice_2�
'simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_4_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02)
'simple_rnn_cell_4/MatMul/ReadVariableOp�
simple_rnn_cell_4/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_4/MatMul�
(simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(simple_rnn_cell_4/BiasAdd/ReadVariableOp�
simple_rnn_cell_4/BiasAddBiasAdd"simple_rnn_cell_4/MatMul:product:00simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_4/BiasAdd�
)simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_4_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype02+
)simple_rnn_cell_4/MatMul_1/ReadVariableOp�
simple_rnn_cell_4/MatMul_1MatMulzeros:output:01simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_4/MatMul_1�
simple_rnn_cell_4/addAddV2"simple_rnn_cell_4/BiasAdd:output:0$simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_4/add�
simple_rnn_cell_4/TanhTanhsimple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:��������� 2
simple_rnn_cell_4/Tanh�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_4_matmul_readvariableop_resource1simple_rnn_cell_4_biasadd_readvariableop_resource2simple_rnn_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :��������� : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_101561*
condR
while_cond_101560*8
output_shapes'
%: : : : :��������� : : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
transpose_1�
IdentityIdentitytranspose_1:y:0)^simple_rnn_cell_4/BiasAdd/ReadVariableOp(^simple_rnn_cell_4/MatMul/ReadVariableOp*^simple_rnn_cell_4/MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :������������������ 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������ :::2T
(simple_rnn_cell_4/BiasAdd/ReadVariableOp(simple_rnn_cell_4/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_4/MatMul/ReadVariableOp'simple_rnn_cell_4/MatMul/ReadVariableOp2V
)simple_rnn_cell_4/MatMul_1/ReadVariableOp)simple_rnn_cell_4/MatMul_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�<
�
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_97937

inputs
simple_rnn_cell_3_97862
simple_rnn_cell_3_97864
simple_rnn_cell_3_97866
identity��)simple_rnn_cell_3/StatefulPartitionedCall�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
)simple_rnn_cell_3/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_3_97862simple_rnn_cell_3_97864simple_rnn_cell_3_97866*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:��������� :��������� *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_simple_rnn_cell_3_layer_call_and_return_conditional_losses_976002+
)simple_rnn_cell_3/StatefulPartitionedCall�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_3_97862simple_rnn_cell_3_97864simple_rnn_cell_3_97866*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :��������� : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_97874*
condR
while_cond_97873*8
output_shapes'
%: : : : :��������� : : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
transpose_1�
IdentityIdentitytranspose_1:y:0*^simple_rnn_cell_3/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :������������������ 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::2V
)simple_rnn_cell_3/StatefulPartitionedCall)simple_rnn_cell_3/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
R
lambda_1_input@
 serving_default_lambda_1_input:0������������������<
dense_230
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�G
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
	optimizer
regularization_losses
		variables

trainable_variables
	keras_api

signatures
+�&call_and_return_all_conditional_losses
�_default_save_signature
�__call__"�D
_tf_keras_sequential�C{"class_name": "Sequential", "name": "sequential_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_1_input"}}, {"class_name": "Lambda", "config": {"name": "lambda_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAEAAAAQwAAAHMOAAAAdABqAXwAZAFkAo0CUwApA07p/////ykB2gRh\neGlzKQLaAnRm2gtleHBhbmRfZGltcykB2gF4qQByBgAAAPofPGlweXRob24taW5wdXQtNjktNTdi\nOWQ0NmZiYWQyPtoIPGxhbWJkYT4FAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}, {"class_name": "SimpleRNN", "config": {"name": "simple_rnn_3", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}, {"class_name": "SimpleRNN", "config": {"name": "simple_rnn_4", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}, {"class_name": "SimpleRNN", "config": {"name": "simple_rnn_5", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}, {"class_name": "Dense", "config": {"name": "dense_22", "trainable": true, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_23", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_1_input"}}, {"class_name": "Lambda", "config": {"name": "lambda_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAEAAAAQwAAAHMOAAAAdABqAXwAZAFkAo0CUwApA07p/////ykB2gRh\neGlzKQLaAnRm2gtleHBhbmRfZGltcykB2gF4qQByBgAAAPofPGlweXRob24taW5wdXQtNjktNTdi\nOWQ0NmZiYWQyPtoIPGxhbWJkYT4FAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}, {"class_name": "SimpleRNN", "config": {"name": "simple_rnn_3", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}, {"class_name": "SimpleRNN", "config": {"name": "simple_rnn_4", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}, {"class_name": "SimpleRNN", "config": {"name": "simple_rnn_5", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}, {"class_name": "Dense", "config": {"name": "dense_22", "trainable": true, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_23", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": {"class_name": "Huber", "config": {"reduction": "auto", "name": "huber_loss", "delta": 1.0}}, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�
regularization_losses
	variables
trainable_variables
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Lambda", "name": "lambda_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "lambda_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAEAAAAQwAAAHMOAAAAdABqAXwAZAFkAo0CUwApA07p/////ykB2gRh\neGlzKQLaAnRm2gtleHBhbmRfZGltcykB2gF4qQByBgAAAPofPGlweXRob24taW5wdXQtNjktNTdi\nOWQ0NmZiYWQyPtoIPGxhbWJkYT4FAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
�

cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_rnn_layer�	{"class_name": "SimpleRNN", "name": "simple_rnn_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "simple_rnn_3", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 1]}}
�

cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_rnn_layer�	{"class_name": "SimpleRNN", "name": "simple_rnn_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "simple_rnn_4", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 32]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 32]}}
�
cell

state_spec
regularization_losses
 	variables
!trainable_variables
"	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_rnn_layer�	{"class_name": "SimpleRNN", "name": "simple_rnn_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "simple_rnn_5", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 32]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 32]}}
�

#kernel
$bias
%regularization_losses
&	variables
'trainable_variables
(	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_22", "trainable": true, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
�

)kernel
*bias
+regularization_losses
,	variables
-trainable_variables
.	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_23", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30]}}
�
/iter

0beta_1

1beta_2
	2decay
3learning_rate#m�$m�)m�*m�4m�5m�6m�7m�8m�9m�:m�;m�<m�#v�$v�)v�*v�4v�5v�6v�7v�8v�9v�:v�;v�<v�"
	optimizer
 "
trackable_list_wrapper
~
40
51
62
73
84
95
:6
;7
<8
#9
$10
)11
*12"
trackable_list_wrapper
~
40
51
62
73
84
95
:6
;7
<8
#9
$10
)11
*12"
trackable_list_wrapper
�
regularization_losses
=layer_metrics
>layer_regularization_losses

?layers
@metrics
Anon_trainable_variables
		variables

trainable_variables
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
regularization_losses
Blayer_metrics
Clayer_regularization_losses
Dmetrics

Elayers
Fnon_trainable_variables
	variables
trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�

4kernel
5recurrent_kernel
6bias
Gregularization_losses
H	variables
Itrainable_variables
J	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "SimpleRNNCell", "name": "simple_rnn_cell_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "simple_rnn_cell_3", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
40
51
62"
trackable_list_wrapper
5
40
51
62"
trackable_list_wrapper
�

Kstates
regularization_losses
Llayer_metrics
Mlayer_regularization_losses
Nmetrics

Olayers
Pnon_trainable_variables
	variables
trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�

7kernel
8recurrent_kernel
9bias
Qregularization_losses
R	variables
Strainable_variables
T	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "SimpleRNNCell", "name": "simple_rnn_cell_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "simple_rnn_cell_4", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
70
81
92"
trackable_list_wrapper
5
70
81
92"
trackable_list_wrapper
�

Ustates
regularization_losses
Vlayer_metrics
Wlayer_regularization_losses
Xmetrics

Ylayers
Znon_trainable_variables
	variables
trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�

:kernel
;recurrent_kernel
<bias
[regularization_losses
\	variables
]trainable_variables
^	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "SimpleRNNCell", "name": "simple_rnn_cell_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "simple_rnn_cell_5", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
:0
;1
<2"
trackable_list_wrapper
5
:0
;1
<2"
trackable_list_wrapper
�

_states
regularization_losses
`layer_metrics
alayer_regularization_losses
bmetrics

clayers
dnon_trainable_variables
 	variables
!trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!: 2dense_22/kernel
:2dense_22/bias
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
�
%regularization_losses
elayer_metrics
flayer_regularization_losses
gmetrics

hlayers
inon_trainable_variables
&	variables
'trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:2dense_23/kernel
:2dense_23/bias
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
�
+regularization_losses
jlayer_metrics
klayer_regularization_losses
lmetrics

mlayers
nnon_trainable_variables
,	variables
-trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
7:5 2%simple_rnn_3/simple_rnn_cell_3/kernel
A:?  2/simple_rnn_3/simple_rnn_cell_3/recurrent_kernel
1:/ 2#simple_rnn_3/simple_rnn_cell_3/bias
7:5  2%simple_rnn_4/simple_rnn_cell_4/kernel
A:?  2/simple_rnn_4/simple_rnn_cell_4/recurrent_kernel
1:/ 2#simple_rnn_4/simple_rnn_cell_4/bias
7:5  2%simple_rnn_5/simple_rnn_cell_5/kernel
A:?  2/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel
1:/ 2#simple_rnn_5/simple_rnn_cell_5/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
'
o0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
40
51
62"
trackable_list_wrapper
5
40
51
62"
trackable_list_wrapper
�
Gregularization_losses
player_metrics
qlayer_regularization_losses
rmetrics

slayers
tnon_trainable_variables
H	variables
Itrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
70
81
92"
trackable_list_wrapper
5
70
81
92"
trackable_list_wrapper
�
Qregularization_losses
ulayer_metrics
vlayer_regularization_losses
wmetrics

xlayers
ynon_trainable_variables
R	variables
Strainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
:0
;1
<2"
trackable_list_wrapper
5
:0
;1
<2"
trackable_list_wrapper
�
[regularization_losses
zlayer_metrics
{layer_regularization_losses
|metrics

}layers
~non_trainable_variables
\	variables
]trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
	total

�count
�	variables
�	keras_api"�
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
/
0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
&:$ 2Adam/dense_22/kernel/m
 :2Adam/dense_22/bias/m
&:$2Adam/dense_23/kernel/m
 :2Adam/dense_23/bias/m
<:: 2,Adam/simple_rnn_3/simple_rnn_cell_3/kernel/m
F:D  26Adam/simple_rnn_3/simple_rnn_cell_3/recurrent_kernel/m
6:4 2*Adam/simple_rnn_3/simple_rnn_cell_3/bias/m
<::  2,Adam/simple_rnn_4/simple_rnn_cell_4/kernel/m
F:D  26Adam/simple_rnn_4/simple_rnn_cell_4/recurrent_kernel/m
6:4 2*Adam/simple_rnn_4/simple_rnn_cell_4/bias/m
<::  2,Adam/simple_rnn_5/simple_rnn_cell_5/kernel/m
F:D  26Adam/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel/m
6:4 2*Adam/simple_rnn_5/simple_rnn_cell_5/bias/m
&:$ 2Adam/dense_22/kernel/v
 :2Adam/dense_22/bias/v
&:$2Adam/dense_23/kernel/v
 :2Adam/dense_23/bias/v
<:: 2,Adam/simple_rnn_3/simple_rnn_cell_3/kernel/v
F:D  26Adam/simple_rnn_3/simple_rnn_cell_3/recurrent_kernel/v
6:4 2*Adam/simple_rnn_3/simple_rnn_cell_3/bias/v
<::  2,Adam/simple_rnn_4/simple_rnn_cell_4/kernel/v
F:D  26Adam/simple_rnn_4/simple_rnn_cell_4/recurrent_kernel/v
6:4 2*Adam/simple_rnn_4/simple_rnn_cell_4/bias/v
<::  2,Adam/simple_rnn_5/simple_rnn_cell_5/kernel/v
F:D  26Adam/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel/v
6:4 2*Adam/simple_rnn_5/simple_rnn_cell_5/bias/v
�2�
G__inference_sequential_6_layer_call_and_return_conditional_losses_99929
H__inference_sequential_6_layer_call_and_return_conditional_losses_100827
H__inference_sequential_6_layer_call_and_return_conditional_losses_100484
G__inference_sequential_6_layer_call_and_return_conditional_losses_99965�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
 __inference__wrapped_model_97551�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *6�3
1�.
lambda_1_input������������������
�2�
-__inference_sequential_6_layer_call_fn_100100
-__inference_sequential_6_layer_call_fn_100889
-__inference_sequential_6_layer_call_fn_100033
-__inference_sequential_6_layer_call_fn_100858�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
D__inference_lambda_1_layer_call_and_return_conditional_losses_100895
D__inference_lambda_1_layer_call_and_return_conditional_losses_100901�
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
)__inference_lambda_1_layer_call_fn_100911
)__inference_lambda_1_layer_call_fn_100906�
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
H__inference_simple_rnn_3_layer_call_and_return_conditional_losses_101269
H__inference_simple_rnn_3_layer_call_and_return_conditional_losses_101023
H__inference_simple_rnn_3_layer_call_and_return_conditional_losses_101135
H__inference_simple_rnn_3_layer_call_and_return_conditional_losses_101381�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
-__inference_simple_rnn_3_layer_call_fn_101146
-__inference_simple_rnn_3_layer_call_fn_101157
-__inference_simple_rnn_3_layer_call_fn_101392
-__inference_simple_rnn_3_layer_call_fn_101403�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_101627
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_101761
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_101873
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_101515�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
-__inference_simple_rnn_4_layer_call_fn_101884
-__inference_simple_rnn_4_layer_call_fn_101895
-__inference_simple_rnn_4_layer_call_fn_101649
-__inference_simple_rnn_4_layer_call_fn_101638�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_102119
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_102365
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_102007
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_102253�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
-__inference_simple_rnn_5_layer_call_fn_102130
-__inference_simple_rnn_5_layer_call_fn_102387
-__inference_simple_rnn_5_layer_call_fn_102141
-__inference_simple_rnn_5_layer_call_fn_102376�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
D__inference_dense_22_layer_call_and_return_conditional_losses_102398�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_dense_22_layer_call_fn_102407�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_dense_23_layer_call_and_return_conditional_losses_102417�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_dense_23_layer_call_fn_102426�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_signature_wrapper_100141lambda_1_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
M__inference_simple_rnn_cell_3_layer_call_and_return_conditional_losses_102443
M__inference_simple_rnn_cell_3_layer_call_and_return_conditional_losses_102460�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
2__inference_simple_rnn_cell_3_layer_call_fn_102474
2__inference_simple_rnn_cell_3_layer_call_fn_102488�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
M__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_102522
M__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_102505�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
2__inference_simple_rnn_cell_4_layer_call_fn_102550
2__inference_simple_rnn_cell_4_layer_call_fn_102536�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
M__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_102567
M__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_102584�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
2__inference_simple_rnn_cell_5_layer_call_fn_102598
2__inference_simple_rnn_cell_5_layer_call_fn_102612�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 �
 __inference__wrapped_model_97551�465798:<;#$)*@�=
6�3
1�.
lambda_1_input������������������
� "3�0
.
dense_23"�
dense_23����������
D__inference_dense_22_layer_call_and_return_conditional_losses_102398\#$/�,
%�"
 �
inputs��������� 
� "%�"
�
0���������
� |
)__inference_dense_22_layer_call_fn_102407O#$/�,
%�"
 �
inputs��������� 
� "�����������
D__inference_dense_23_layer_call_and_return_conditional_losses_102417\)*/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� |
)__inference_dense_23_layer_call_fn_102426O)*/�,
%�"
 �
inputs���������
� "�����������
D__inference_lambda_1_layer_call_and_return_conditional_losses_100895v@�=
6�3
)�&
inputs������������������

 
p
� "2�/
(�%
0������������������
� �
D__inference_lambda_1_layer_call_and_return_conditional_losses_100901v@�=
6�3
)�&
inputs������������������

 
p 
� "2�/
(�%
0������������������
� �
)__inference_lambda_1_layer_call_fn_100906i@�=
6�3
)�&
inputs������������������

 
p
� "%�"�������������������
)__inference_lambda_1_layer_call_fn_100911i@�=
6�3
)�&
inputs������������������

 
p 
� "%�"�������������������
H__inference_sequential_6_layer_call_and_return_conditional_losses_100484x465798:<;#$)*@�=
6�3
)�&
inputs������������������
p

 
� "%�"
�
0���������
� �
H__inference_sequential_6_layer_call_and_return_conditional_losses_100827x465798:<;#$)*@�=
6�3
)�&
inputs������������������
p 

 
� "%�"
�
0���������
� �
G__inference_sequential_6_layer_call_and_return_conditional_losses_99929�465798:<;#$)*H�E
>�;
1�.
lambda_1_input������������������
p

 
� "%�"
�
0���������
� �
G__inference_sequential_6_layer_call_and_return_conditional_losses_99965�465798:<;#$)*H�E
>�;
1�.
lambda_1_input������������������
p 

 
� "%�"
�
0���������
� �
-__inference_sequential_6_layer_call_fn_100033s465798:<;#$)*H�E
>�;
1�.
lambda_1_input������������������
p

 
� "�����������
-__inference_sequential_6_layer_call_fn_100100s465798:<;#$)*H�E
>�;
1�.
lambda_1_input������������������
p 

 
� "�����������
-__inference_sequential_6_layer_call_fn_100858k465798:<;#$)*@�=
6�3
)�&
inputs������������������
p

 
� "�����������
-__inference_sequential_6_layer_call_fn_100889k465798:<;#$)*@�=
6�3
)�&
inputs������������������
p 

 
� "�����������
$__inference_signature_wrapper_100141�465798:<;#$)*R�O
� 
H�E
C
lambda_1_input1�.
lambda_1_input������������������"3�0
.
dense_23"�
dense_23����������
H__inference_simple_rnn_3_layer_call_and_return_conditional_losses_101023�465O�L
E�B
4�1
/�,
inputs/0������������������

 
p

 
� "2�/
(�%
0������������������ 
� �
H__inference_simple_rnn_3_layer_call_and_return_conditional_losses_101135�465O�L
E�B
4�1
/�,
inputs/0������������������

 
p 

 
� "2�/
(�%
0������������������ 
� �
H__inference_simple_rnn_3_layer_call_and_return_conditional_losses_101269�465H�E
>�;
-�*
inputs������������������

 
p

 
� "2�/
(�%
0������������������ 
� �
H__inference_simple_rnn_3_layer_call_and_return_conditional_losses_101381�465H�E
>�;
-�*
inputs������������������

 
p 

 
� "2�/
(�%
0������������������ 
� �
-__inference_simple_rnn_3_layer_call_fn_101146}465O�L
E�B
4�1
/�,
inputs/0������������������

 
p

 
� "%�"������������������ �
-__inference_simple_rnn_3_layer_call_fn_101157}465O�L
E�B
4�1
/�,
inputs/0������������������

 
p 

 
� "%�"������������������ �
-__inference_simple_rnn_3_layer_call_fn_101392v465H�E
>�;
-�*
inputs������������������

 
p

 
� "%�"������������������ �
-__inference_simple_rnn_3_layer_call_fn_101403v465H�E
>�;
-�*
inputs������������������

 
p 

 
� "%�"������������������ �
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_101515�798H�E
>�;
-�*
inputs������������������ 

 
p

 
� "2�/
(�%
0������������������ 
� �
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_101627�798H�E
>�;
-�*
inputs������������������ 

 
p 

 
� "2�/
(�%
0������������������ 
� �
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_101761�798O�L
E�B
4�1
/�,
inputs/0������������������ 

 
p

 
� "2�/
(�%
0������������������ 
� �
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_101873�798O�L
E�B
4�1
/�,
inputs/0������������������ 

 
p 

 
� "2�/
(�%
0������������������ 
� �
-__inference_simple_rnn_4_layer_call_fn_101638v798H�E
>�;
-�*
inputs������������������ 

 
p

 
� "%�"������������������ �
-__inference_simple_rnn_4_layer_call_fn_101649v798H�E
>�;
-�*
inputs������������������ 

 
p 

 
� "%�"������������������ �
-__inference_simple_rnn_4_layer_call_fn_101884}798O�L
E�B
4�1
/�,
inputs/0������������������ 

 
p

 
� "%�"������������������ �
-__inference_simple_rnn_4_layer_call_fn_101895}798O�L
E�B
4�1
/�,
inputs/0������������������ 

 
p 

 
� "%�"������������������ �
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_102007}:<;O�L
E�B
4�1
/�,
inputs/0������������������ 

 
p

 
� "%�"
�
0��������� 
� �
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_102119}:<;O�L
E�B
4�1
/�,
inputs/0������������������ 

 
p 

 
� "%�"
�
0��������� 
� �
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_102253v:<;H�E
>�;
-�*
inputs������������������ 

 
p

 
� "%�"
�
0��������� 
� �
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_102365v:<;H�E
>�;
-�*
inputs������������������ 

 
p 

 
� "%�"
�
0��������� 
� �
-__inference_simple_rnn_5_layer_call_fn_102130p:<;O�L
E�B
4�1
/�,
inputs/0������������������ 

 
p

 
� "���������� �
-__inference_simple_rnn_5_layer_call_fn_102141p:<;O�L
E�B
4�1
/�,
inputs/0������������������ 

 
p 

 
� "���������� �
-__inference_simple_rnn_5_layer_call_fn_102376i:<;H�E
>�;
-�*
inputs������������������ 

 
p

 
� "���������� �
-__inference_simple_rnn_5_layer_call_fn_102387i:<;H�E
>�;
-�*
inputs������������������ 

 
p 

 
� "���������� �
M__inference_simple_rnn_cell_3_layer_call_and_return_conditional_losses_102443�465\�Y
R�O
 �
inputs���������
'�$
"�
states/0��������� 
p
� "R�O
H�E
�
0/0��������� 
$�!
�
0/1/0��������� 
� �
M__inference_simple_rnn_cell_3_layer_call_and_return_conditional_losses_102460�465\�Y
R�O
 �
inputs���������
'�$
"�
states/0��������� 
p 
� "R�O
H�E
�
0/0��������� 
$�!
�
0/1/0��������� 
� �
2__inference_simple_rnn_cell_3_layer_call_fn_102474�465\�Y
R�O
 �
inputs���������
'�$
"�
states/0��������� 
p
� "D�A
�
0��������� 
"�
�
1/0��������� �
2__inference_simple_rnn_cell_3_layer_call_fn_102488�465\�Y
R�O
 �
inputs���������
'�$
"�
states/0��������� 
p 
� "D�A
�
0��������� 
"�
�
1/0��������� �
M__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_102505�798\�Y
R�O
 �
inputs��������� 
'�$
"�
states/0��������� 
p
� "R�O
H�E
�
0/0��������� 
$�!
�
0/1/0��������� 
� �
M__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_102522�798\�Y
R�O
 �
inputs��������� 
'�$
"�
states/0��������� 
p 
� "R�O
H�E
�
0/0��������� 
$�!
�
0/1/0��������� 
� �
2__inference_simple_rnn_cell_4_layer_call_fn_102536�798\�Y
R�O
 �
inputs��������� 
'�$
"�
states/0��������� 
p
� "D�A
�
0��������� 
"�
�
1/0��������� �
2__inference_simple_rnn_cell_4_layer_call_fn_102550�798\�Y
R�O
 �
inputs��������� 
'�$
"�
states/0��������� 
p 
� "D�A
�
0��������� 
"�
�
1/0��������� �
M__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_102567�:<;\�Y
R�O
 �
inputs��������� 
'�$
"�
states/0��������� 
p
� "R�O
H�E
�
0/0��������� 
$�!
�
0/1/0��������� 
� �
M__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_102584�:<;\�Y
R�O
 �
inputs��������� 
'�$
"�
states/0��������� 
p 
� "R�O
H�E
�
0/0��������� 
$�!
�
0/1/0��������� 
� �
2__inference_simple_rnn_cell_5_layer_call_fn_102598�:<;\�Y
R�O
 �
inputs��������� 
'�$
"�
states/0��������� 
p
� "D�A
�
0��������� 
"�
�
1/0��������� �
2__inference_simple_rnn_cell_5_layer_call_fn_102612�:<;\�Y
R�O
 �
inputs��������� 
'�$
"�
states/0��������� 
p 
� "D�A
�
0��������� 
"�
�
1/0��������� 