
�
I%sTime (s): cpu = %s ; elapsed = %s . Memory (MB): peak = %s ; gain = %s
268*common2
create_project: 2

00:00:062

00:00:082	
567.1842	
163.844Z17-268h px� 
>
Refreshing IP repositories
234*coregenZ19-234h px� 
�
 Loaded user IP repository '%s'.
1135*coregen2L
Jc:/Main_Folder/School/Year_4/Capstone/vitis_testing/hls/matmul_partitionedZ19-1700h px� 
j
"Loaded Vivado IP repository '%s'.
1332*coregen2!
D:/Vivado/Vivado/2024.2/data/ipZ19-2313h px� 
�
�Found utility IPs instantiated in block design %s which have equivalent inline hdl with improved performance and reduced diskspace.
It is recommended to migrate these utility IPs to inline hdl using the command upgrade_project -migrate_to_inline_hdl.  The utility IPs may be deprecated in future releases.
More information on inline hdl is available in UG994. 
28820*project2k
iC:/Main_Folder/School/Year_4/Capstone/pynq/pynq_test/pynq_test.srcs/sources_1/bd/matmult_ex/matmult_ex.bdZ1-5578h px� 
�
I%sTime (s): cpu = %s ; elapsed = %s . Memory (MB): peak = %s ; gain = %s
268*common2
add_files: 2

00:00:032

00:00:062	
613.5702
46.387Z17-268h px� 
�
Command: %s
1870*	planAhead2�
�read_checkpoint -auto_incremental -incremental C:/Main_Folder/School/Year_4/Capstone/pynq/pynq_test/pynq_test.srcs/utils_1/imports/synth_1/design_1_wrapper.dcpZ12-2866h px� 
�
;Read reference checkpoint from %s for incremental synthesis3154*	planAhead2r
pC:/Main_Folder/School/Year_4/Capstone/pynq/pynq_test/pynq_test.srcs/utils_1/imports/synth_1/design_1_wrapper.dcpZ12-5825h px� 
T
-Please ensure there are no constraint changes3725*	planAheadZ12-7989h px� 
m
Command: %s
53*	vivadotcl2<
:synth_design -top matmult_ex_wrapper -part xc7z020clg400-1Z4-113h px� 
:
Starting synth_design
149*	vivadotclZ4-321h px� 
z
@Attempting to get a license for feature '%s' and/or device '%s'
308*common2
	Synthesis2	
xc7z020Z17-347h px� 
j
0Got license for feature '%s' and/or device '%s'
310*common2
	Synthesis2	
xc7z020Z17-349h px� 
D
Loading part %s157*device2
xc7z020clg400-1Z21-403h px� 

VNo compile time benefit to using incremental synthesis; A full resynthesis will be run2353*designutilsZ20-5440h px� 
�
�Flow is switching to default flow due to incremental criteria not met. If you would like to alter this behaviour and have the flow terminate instead, please set the following parameter config_implementation {autoIncr.Synth.RejectBehavior Terminate}2229*designutilsZ20-4379h px� 
o
HMultithreading enabled for synth_design using a maximum of %s processes.4828*oasys2
2Z8-7079h px� 
a
?Launching helper process for spawning children vivado processes4827*oasysZ8-7078h px� 
N
#Helper process launched with PID %s4824*oasys2
22404Z8-7075h px� 
�
%s*synth2{
yStarting RTL Elaboration : Time (s): cpu = 00:00:05 ; elapsed = 00:00:08 . Memory (MB): peak = 1316.145 ; gain = 469.246
h px� 
�
synthesizing module '%s'%s4497*oasys2
matmult_ex_wrapper2
 2w
sc:/Main_Folder/School/Year_4/Capstone/pynq/pynq_test/pynq_test.gen/sources_1/bd/matmult_ex/hdl/matmult_ex_wrapper.v2
138@Z8-6157h px� 
�
synthesizing module '%s'%s4497*oasys2

matmult_ex2
 2q
mc:/Main_Folder/School/Year_4/Capstone/pynq/pynq_test/pynq_test.gen/sources_1/bd/matmult_ex/synth/matmult_ex.v2
138@Z8-6157h px� 
�
synthesizing module '%s'%s4497*oasys2
matmult_ex_axi_smc_02
 2�
�C:/Main_Folder/School/Year_4/Capstone/pynq/pynq_test/pynq_test.runs/synth_1/.Xil/Vivado-16516-TABLET-60QL2L1F/realtime/matmult_ex_axi_smc_0_stub.v2
68@Z8-6157h px� 
�
'done synthesizing module '%s'%s (%s#%s)4495*oasys2
matmult_ex_axi_smc_02
 2
02
12�
�C:/Main_Folder/School/Year_4/Capstone/pynq/pynq_test/pynq_test.runs/synth_1/.Xil/Vivado-16516-TABLET-60QL2L1F/realtime/matmult_ex_axi_smc_0_stub.v2
68@Z8-6155h px� 
�
9port '%s' of module '%s' is unconnected for instance '%s'4818*oasys2
M00_AXI_awprot2
matmult_ex_axi_smc_02	
axi_smc2q
mc:/Main_Folder/School/Year_4/Capstone/pynq/pynq_test/pynq_test.gen/sources_1/bd/matmult_ex/synth/matmult_ex.v2
1778@Z8-7071h px� 
�
9port '%s' of module '%s' is unconnected for instance '%s'4818*oasys2
M00_AXI_arprot2
matmult_ex_axi_smc_02	
axi_smc2q
mc:/Main_Folder/School/Year_4/Capstone/pynq/pynq_test/pynq_test.gen/sources_1/bd/matmult_ex/synth/matmult_ex.v2
1778@Z8-7071h px� 
�
Kinstance '%s' of module '%s' has %s connections declared, but only %s given4757*oasys2	
axi_smc2
matmult_ex_axi_smc_02
962
942q
mc:/Main_Folder/School/Year_4/Capstone/pynq/pynq_test/pynq_test.gen/sources_1/bd/matmult_ex/synth/matmult_ex.v2
1778@Z8-7023h px� 
�
synthesizing module '%s'%s4497*oasys2!
matmult_ex_matmul_partition_0_02
 2�
�C:/Main_Folder/School/Year_4/Capstone/pynq/pynq_test/pynq_test.runs/synth_1/.Xil/Vivado-16516-TABLET-60QL2L1F/realtime/matmult_ex_matmul_partition_0_0_stub.v2
68@Z8-6157h px� 
�
'done synthesizing module '%s'%s (%s#%s)4495*oasys2!
matmult_ex_matmul_partition_0_02
 2
02
12�
�C:/Main_Folder/School/Year_4/Capstone/pynq/pynq_test/pynq_test.runs/synth_1/.Xil/Vivado-16516-TABLET-60QL2L1F/realtime/matmult_ex_matmul_partition_0_0_stub.v2
68@Z8-6155h px� 
�
9port '%s' of module '%s' is unconnected for instance '%s'4818*oasys2
	interrupt2!
matmult_ex_matmul_partition_0_02
matmul_partition_02q
mc:/Main_Folder/School/Year_4/Capstone/pynq/pynq_test/pynq_test.gen/sources_1/bd/matmult_ex/synth/matmult_ex.v2
2728@Z8-7071h px� 
�
9port '%s' of module '%s' is unconnected for instance '%s'4818*oasys2
m_axi_gmem_ARREGION2!
matmult_ex_matmul_partition_0_02
matmul_partition_02q
mc:/Main_Folder/School/Year_4/Capstone/pynq/pynq_test/pynq_test.gen/sources_1/bd/matmult_ex/synth/matmult_ex.v2
2728@Z8-7071h px� 
�
9port '%s' of module '%s' is unconnected for instance '%s'4818*oasys2
m_axi_gmem_AWREGION2!
matmult_ex_matmul_partition_0_02
matmul_partition_02q
mc:/Main_Folder/School/Year_4/Capstone/pynq/pynq_test/pynq_test.gen/sources_1/bd/matmult_ex/synth/matmult_ex.v2
2728@Z8-7071h px� 
�
9port '%s' of module '%s' is unconnected for instance '%s'4818*oasys2
m_axi_gmem_WID2!
matmult_ex_matmul_partition_0_02
matmul_partition_02q
mc:/Main_Folder/School/Year_4/Capstone/pynq/pynq_test/pynq_test.gen/sources_1/bd/matmult_ex/synth/matmult_ex.v2
2728@Z8-7071h px� 
�
Kinstance '%s' of module '%s' has %s connections declared, but only %s given4757*oasys2
matmul_partition_02!
matmult_ex_matmul_partition_0_02
622
582q
mc:/Main_Folder/School/Year_4/Capstone/pynq/pynq_test/pynq_test.gen/sources_1/bd/matmult_ex/synth/matmult_ex.v2
2728@Z8-7023h px� 
�
synthesizing module '%s'%s4497*oasys2#
!matmult_ex_processing_system7_0_02
 2�
�C:/Main_Folder/School/Year_4/Capstone/pynq/pynq_test/pynq_test.runs/synth_1/.Xil/Vivado-16516-TABLET-60QL2L1F/realtime/matmult_ex_processing_system7_0_0_stub.v2
68@Z8-6157h px� 
�
'done synthesizing module '%s'%s (%s#%s)4495*oasys2#
!matmult_ex_processing_system7_0_02
 2
02
12�
�C:/Main_Folder/School/Year_4/Capstone/pynq/pynq_test/pynq_test.runs/synth_1/.Xil/Vivado-16516-TABLET-60QL2L1F/realtime/matmult_ex_processing_system7_0_0_stub.v2
68@Z8-6155h px� 
�
9port '%s' of module '%s' is unconnected for instance '%s'4818*oasys2
USB0_PORT_INDCTL2#
!matmult_ex_processing_system7_0_02
processing_system7_02q
mc:/Main_Folder/School/Year_4/Capstone/pynq/pynq_test/pynq_test.gen/sources_1/bd/matmult_ex/synth/matmult_ex.v2
3318@Z8-7071h px� 
�
9port '%s' of module '%s' is unconnected for instance '%s'4818*oasys2
USB0_VBUS_PWRSELECT2#
!matmult_ex_processing_system7_0_02
processing_system7_02q
mc:/Main_Folder/School/Year_4/Capstone/pynq/pynq_test/pynq_test.gen/sources_1/bd/matmult_ex/synth/matmult_ex.v2
3318@Z8-7071h px� 
�
Kinstance '%s' of module '%s' has %s connections declared, but only %s given4757*oasys2
processing_system7_02#
!matmult_ex_processing_system7_0_02
652
632q
mc:/Main_Folder/School/Year_4/Capstone/pynq/pynq_test/pynq_test.gen/sources_1/bd/matmult_ex/synth/matmult_ex.v2
3318@Z8-7023h px� 
�
synthesizing module '%s'%s4497*oasys2
matmult_ex_rst_ps7_0_100M_02
 2�
�C:/Main_Folder/School/Year_4/Capstone/pynq/pynq_test/pynq_test.runs/synth_1/.Xil/Vivado-16516-TABLET-60QL2L1F/realtime/matmult_ex_rst_ps7_0_100M_0_stub.v2
68@Z8-6157h px� 
�
'done synthesizing module '%s'%s (%s#%s)4495*oasys2
matmult_ex_rst_ps7_0_100M_02
 2
02
12�
�C:/Main_Folder/School/Year_4/Capstone/pynq/pynq_test/pynq_test.runs/synth_1/.Xil/Vivado-16516-TABLET-60QL2L1F/realtime/matmult_ex_rst_ps7_0_100M_0_stub.v2
68@Z8-6155h px� 
�
9port '%s' of module '%s' is unconnected for instance '%s'4818*oasys2

mb_reset2
matmult_ex_rst_ps7_0_100M_02
rst_ps7_0_100M2q
mc:/Main_Folder/School/Year_4/Capstone/pynq/pynq_test/pynq_test.gen/sources_1/bd/matmult_ex/synth/matmult_ex.v2
3958@Z8-7071h px� 
�
9port '%s' of module '%s' is unconnected for instance '%s'4818*oasys2
bus_struct_reset2
matmult_ex_rst_ps7_0_100M_02
rst_ps7_0_100M2q
mc:/Main_Folder/School/Year_4/Capstone/pynq/pynq_test/pynq_test.gen/sources_1/bd/matmult_ex/synth/matmult_ex.v2
3958@Z8-7071h px� 
�
9port '%s' of module '%s' is unconnected for instance '%s'4818*oasys2
peripheral_reset2
matmult_ex_rst_ps7_0_100M_02
rst_ps7_0_100M2q
mc:/Main_Folder/School/Year_4/Capstone/pynq/pynq_test/pynq_test.gen/sources_1/bd/matmult_ex/synth/matmult_ex.v2
3958@Z8-7071h px� 
�
9port '%s' of module '%s' is unconnected for instance '%s'4818*oasys2
interconnect_aresetn2
matmult_ex_rst_ps7_0_100M_02
rst_ps7_0_100M2q
mc:/Main_Folder/School/Year_4/Capstone/pynq/pynq_test/pynq_test.gen/sources_1/bd/matmult_ex/synth/matmult_ex.v2
3958@Z8-7071h px� 
�
Kinstance '%s' of module '%s' has %s connections declared, but only %s given4757*oasys2
rst_ps7_0_100M2
matmult_ex_rst_ps7_0_100M_02
102
62q
mc:/Main_Folder/School/Year_4/Capstone/pynq/pynq_test/pynq_test.gen/sources_1/bd/matmult_ex/synth/matmult_ex.v2
3958@Z8-7023h px� 
�
synthesizing module '%s'%s4497*oasys2
matmult_ex_xlconstant_0_02
 2�
�c:/Main_Folder/School/Year_4/Capstone/pynq/pynq_test/pynq_test.gen/sources_1/bd/matmult_ex/ip/matmult_ex_xlconstant_0_0/synth/matmult_ex_xlconstant_0_0.v2
538@Z8-6157h px� 
�
synthesizing module '%s'%s4497*oasys2
xlconstant_v1_1_9_xlconstant2
 2�
�c:/Main_Folder/School/Year_4/Capstone/pynq/pynq_test/pynq_test.gen/sources_1/bd/matmult_ex/ipshared/e2d2/hdl/xlconstant_v1_1_vl_rfs.v2
688@Z8-6157h px� 
�
'done synthesizing module '%s'%s (%s#%s)4495*oasys2
xlconstant_v1_1_9_xlconstant2
 2
02
12�
�c:/Main_Folder/School/Year_4/Capstone/pynq/pynq_test/pynq_test.gen/sources_1/bd/matmult_ex/ipshared/e2d2/hdl/xlconstant_v1_1_vl_rfs.v2
688@Z8-6155h px� 
�
'done synthesizing module '%s'%s (%s#%s)4495*oasys2
matmult_ex_xlconstant_0_02
 2
02
12�
�c:/Main_Folder/School/Year_4/Capstone/pynq/pynq_test/pynq_test.gen/sources_1/bd/matmult_ex/ip/matmult_ex_xlconstant_0_0/synth/matmult_ex_xlconstant_0_0.v2
538@Z8-6155h px� 
�
synthesizing module '%s'%s4497*oasys2
matmult_ex_xlconstant_0_12
 2�
�c:/Main_Folder/School/Year_4/Capstone/pynq/pynq_test/pynq_test.gen/sources_1/bd/matmult_ex/ip/matmult_ex_xlconstant_0_1/synth/matmult_ex_xlconstant_0_1.v2
538@Z8-6157h px� 
�
'done synthesizing module '%s'%s (%s#%s)4495*oasys2
matmult_ex_xlconstant_0_12
 2
02
12�
�c:/Main_Folder/School/Year_4/Capstone/pynq/pynq_test/pynq_test.gen/sources_1/bd/matmult_ex/ip/matmult_ex_xlconstant_0_1/synth/matmult_ex_xlconstant_0_1.v2
538@Z8-6155h px� 
�
'done synthesizing module '%s'%s (%s#%s)4495*oasys2

matmult_ex2
 2
02
12q
mc:/Main_Folder/School/Year_4/Capstone/pynq/pynq_test/pynq_test.gen/sources_1/bd/matmult_ex/synth/matmult_ex.v2
138@Z8-6155h px� 
�
'done synthesizing module '%s'%s (%s#%s)4495*oasys2
matmult_ex_wrapper2
 2
02
12w
sc:/Main_Folder/School/Year_4/Capstone/pynq/pynq_test/pynq_test.gen/sources_1/bd/matmult_ex/hdl/matmult_ex_wrapper.v2
138@Z8-6155h px� 
�
%s*synth2{
yFinished RTL Elaboration : Time (s): cpu = 00:00:07 ; elapsed = 00:00:11 . Memory (MB): peak = 1428.098 ; gain = 581.199
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
;
%s
*synth2#
!Start Handling Custom Attributes
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2�
�Finished Handling Custom Attributes : Time (s): cpu = 00:00:07 ; elapsed = 00:00:11 . Memory (MB): peak = 1428.098 ; gain = 581.199
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2�
�Finished RTL Optimization Phase 1 : Time (s): cpu = 00:00:07 ; elapsed = 00:00:11 . Memory (MB): peak = 1428.098 ; gain = 581.199
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
I%sTime (s): cpu = %s ; elapsed = %s . Memory (MB): peak = %s ; gain = %s
268*common2
Netlist sorting complete. 2

00:00:002
00:00:00.0102

1428.0982
0.000Z17-268h px� 
K
)Preparing netlist for logic optimization
349*projectZ1-570h px� 
>

Processing XDC Constraints
244*projectZ1-262h px� 
=
Initializing timing engine
348*projectZ1-569h px� 
�
$Parsing XDC File [%s] for cell '%s'
848*designutils2�
�c:/Main_Folder/School/Year_4/Capstone/pynq/pynq_test/pynq_test.gen/sources_1/bd/matmult_ex/ip/matmult_ex_processing_system7_0_0/matmult_ex_processing_system7_0_0/matmult_ex_processing_system7_0_0_in_context.xdc2%
!matmult_ex_i/processing_system7_0	8Z20-848h px� 
�
-Finished Parsing XDC File [%s] for cell '%s'
847*designutils2�
�c:/Main_Folder/School/Year_4/Capstone/pynq/pynq_test/pynq_test.gen/sources_1/bd/matmult_ex/ip/matmult_ex_processing_system7_0_0/matmult_ex_processing_system7_0_0/matmult_ex_processing_system7_0_0_in_context.xdc2%
!matmult_ex_i/processing_system7_0	8Z20-847h px� 
�
$Parsing XDC File [%s] for cell '%s'
848*designutils2�
�c:/Main_Folder/School/Year_4/Capstone/pynq/pynq_test/pynq_test.gen/sources_1/bd/matmult_ex/ip/matmult_ex_matmul_partition_0_0/matmult_ex_matmul_partition_0_0/matmult_ex_matmul_partition_0_0_in_context.xdc2#
matmult_ex_i/matmul_partition_0	8Z20-848h px� 
�
-Finished Parsing XDC File [%s] for cell '%s'
847*designutils2�
�c:/Main_Folder/School/Year_4/Capstone/pynq/pynq_test/pynq_test.gen/sources_1/bd/matmult_ex/ip/matmult_ex_matmul_partition_0_0/matmult_ex_matmul_partition_0_0/matmult_ex_matmul_partition_0_0_in_context.xdc2#
matmult_ex_i/matmul_partition_0	8Z20-847h px� 
�
$Parsing XDC File [%s] for cell '%s'
848*designutils2�
�c:/Main_Folder/School/Year_4/Capstone/pynq/pynq_test/pynq_test.gen/sources_1/bd/matmult_ex/ip/matmult_ex_axi_smc_0/matmult_ex_axi_smc_0/matmult_ex_axi_smc_0_in_context.xdc2
matmult_ex_i/axi_smc	8Z20-848h px� 
�
-Finished Parsing XDC File [%s] for cell '%s'
847*designutils2�
�c:/Main_Folder/School/Year_4/Capstone/pynq/pynq_test/pynq_test.gen/sources_1/bd/matmult_ex/ip/matmult_ex_axi_smc_0/matmult_ex_axi_smc_0/matmult_ex_axi_smc_0_in_context.xdc2
matmult_ex_i/axi_smc	8Z20-847h px� 
�
$Parsing XDC File [%s] for cell '%s'
848*designutils2�
�c:/Main_Folder/School/Year_4/Capstone/pynq/pynq_test/pynq_test.gen/sources_1/bd/matmult_ex/ip/matmult_ex_rst_ps7_0_100M_0/matmult_ex_rst_ps7_0_100M_0/matmult_ex_rst_ps7_0_100M_0_in_context.xdc2
matmult_ex_i/rst_ps7_0_100M	8Z20-848h px� 
�
-Finished Parsing XDC File [%s] for cell '%s'
847*designutils2�
�c:/Main_Folder/School/Year_4/Capstone/pynq/pynq_test/pynq_test.gen/sources_1/bd/matmult_ex/ip/matmult_ex_rst_ps7_0_100M_0/matmult_ex_rst_ps7_0_100M_0/matmult_ex_rst_ps7_0_100M_0_in_context.xdc2
matmult_ex_i/rst_ps7_0_100M	8Z20-847h px� 
�
Parsing XDC File [%s]
179*designutils2?
;C:/Main_Folder/School/Year_4/Capstone/pynq/PYNQ-Z2 v1.0.xdc8Z20-179h px� 
�
No ports matched '%s'.
584*	planAhead2
led_tri_io[0]2?
;C:/Main_Folder/School/Year_4/Capstone/pynq/PYNQ-Z2 v1.0.xdc2
278@Z12-584h px�
�
No ports matched '%s'.
584*	planAhead2
led_tri_io[1]2?
;C:/Main_Folder/School/Year_4/Capstone/pynq/PYNQ-Z2 v1.0.xdc2
288@Z12-584h px�
�
No ports matched '%s'.
584*	planAhead2
led_tri_io[2]2?
;C:/Main_Folder/School/Year_4/Capstone/pynq/PYNQ-Z2 v1.0.xdc2
298@Z12-584h px�
�
No ports matched '%s'.
584*	planAhead2
led_tri_io[3]2?
;C:/Main_Folder/School/Year_4/Capstone/pynq/PYNQ-Z2 v1.0.xdc2
308@Z12-584h px�
�
No ports matched '%s'.
584*	planAhead2
btn_tri_i[0]2?
;C:/Main_Folder/School/Year_4/Capstone/pynq/PYNQ-Z2 v1.0.xdc2
348@Z12-584h px�
�
No ports matched '%s'.
584*	planAhead2
btn_tri_i[1]2?
;C:/Main_Folder/School/Year_4/Capstone/pynq/PYNQ-Z2 v1.0.xdc2
358@Z12-584h px�
�
No ports matched '%s'.
584*	planAhead2
btn_tri_i[2]2?
;C:/Main_Folder/School/Year_4/Capstone/pynq/PYNQ-Z2 v1.0.xdc2
368@Z12-584h px�
�
No ports matched '%s'.
584*	planAhead2
btn_tri_i[3]2?
;C:/Main_Folder/School/Year_4/Capstone/pynq/PYNQ-Z2 v1.0.xdc2
378@Z12-584h px�
�
Finished Parsing XDC File [%s]
178*designutils2?
;C:/Main_Folder/School/Year_4/Capstone/pynq/PYNQ-Z2 v1.0.xdc8Z20-178h px� 
�
�One or more constraints failed evaluation while reading constraint file [%s] and the design contains unresolved black boxes. These constraints will be read post-synthesis (as long as their source constraint file is marked as used_in_implementation) and should be applied correctly then. You should review the constraints listed in the file [%s] and check the run log file to verify that these constraints were correctly applied.301*project2=
;C:/Main_Folder/School/Year_4/Capstone/pynq/PYNQ-Z2 v1.0.xdc2&
$.Xil/matmult_ex_wrapper_propImpl.xdcZ1-498h px� 
�
Parsing XDC File [%s]
179*designutils2^
ZC:/Main_Folder/School/Year_4/Capstone/pynq/pynq_test/pynq_test.runs/synth_1/dont_touch.xdc8Z20-179h px� 
�
Finished Parsing XDC File [%s]
178*designutils2^
ZC:/Main_Folder/School/Year_4/Capstone/pynq/pynq_test/pynq_test.runs/synth_1/dont_touch.xdc8Z20-178h px� 
H
&Completed Processing XDC Constraints

245*projectZ1-263h px� 
�
I%sTime (s): cpu = %s ; elapsed = %s . Memory (MB): peak = %s ; gain = %s
268*common2
Netlist sorting complete. 2

00:00:002

00:00:002

1433.3872
0.000Z17-268h px� 
l
!Unisim Transformation Summary:
%s111*project2'
%No Unisim elements were transformed.
Z1-111h px� 
�
I%sTime (s): cpu = %s ; elapsed = %s . Memory (MB): peak = %s ; gain = %s
268*common2"
 Constraint Validation Runtime : 2

00:00:002
00:00:00.0092

1433.3872
0.000Z17-268h px� 

VNo compile time benefit to using incremental synthesis; A full resynthesis will be run2353*designutilsZ20-5440h px� 
�
�Flow is switching to default flow due to incremental criteria not met. If you would like to alter this behaviour and have the flow terminate instead, please set the following parameter config_implementation {autoIncr.Synth.RejectBehavior Terminate}2229*designutilsZ20-4379h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2�
Finished Constraint Validation : Time (s): cpu = 00:00:16 ; elapsed = 00:00:23 . Memory (MB): peak = 1433.387 ; gain = 586.488
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
D
%s
*synth2,
*Start Loading Part and Timing Information
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
8
%s
*synth2 
Loading part: xc7z020clg400-1
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2�
�Finished Loading Part and Timing Information : Time (s): cpu = 00:00:16 ; elapsed = 00:00:23 . Memory (MB): peak = 1433.387 ; gain = 586.488
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
H
%s
*synth20
.Start Applying 'set_property' XDC Constraints
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2�
�Finished applying 'set_property' XDC Constraints : Time (s): cpu = 00:00:17 ; elapsed = 00:00:23 . Memory (MB): peak = 1433.387 ; gain = 586.488
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2�
�Finished RTL Optimization Phase 2 : Time (s): cpu = 00:00:17 ; elapsed = 00:00:23 . Memory (MB): peak = 1433.387 ; gain = 586.488
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
:
%s
*synth2"
 Start RTL Component Statistics 
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
9
%s
*synth2!
Detailed RTL Component Info : 
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
=
%s
*synth2%
#Finished RTL Component Statistics 
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
6
%s
*synth2
Start Part Resource Summary
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
q
%s
*synth2Y
WPart Resources:
DSPs: 220 (col length:60)
BRAMs: 280 (col length: RAMB18 60 RAMB36 30)
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
9
%s
*synth2!
Finished Part Resource Summary
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
E
%s
*synth2-
+Start Cross Boundary and Area Optimization
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
H
&Parallel synthesis criteria is not met4829*oasysZ8-7080h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2�
�Finished Cross Boundary and Area Optimization : Time (s): cpu = 00:00:19 ; elapsed = 00:00:26 . Memory (MB): peak = 1433.387 ; gain = 586.488
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
@
%s
*synth2(
&Start Applying XDC Timing Constraints
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2�
�Finished Applying XDC Timing Constraints : Time (s): cpu = 00:00:26 ; elapsed = 00:00:35 . Memory (MB): peak = 1607.594 ; gain = 760.695
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
4
%s
*synth2
Start Timing Optimization
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2
}Finished Timing Optimization : Time (s): cpu = 00:00:26 ; elapsed = 00:00:35 . Memory (MB): peak = 1607.746 ; gain = 760.848
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
3
%s
*synth2
Start Technology Mapping
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2~
|Finished Technology Mapping : Time (s): cpu = 00:00:26 ; elapsed = 00:00:35 . Memory (MB): peak = 1618.137 ; gain = 771.238
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
-
%s
*synth2
Start IO Insertion
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
?
%s
*synth2'
%Start Flattening Before IO Insertion
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
B
%s
*synth2*
(Finished Flattening Before IO Insertion
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
6
%s
*synth2
Start Final Netlist Cleanup
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
9
%s
*synth2!
Finished Final Netlist Cleanup
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2x
vFinished IO Insertion : Time (s): cpu = 00:00:30 ; elapsed = 00:00:41 . Memory (MB): peak = 1834.375 ; gain = 987.477
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
=
%s
*synth2%
#Start Renaming Generated Instances
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2�
�Finished Renaming Generated Instances : Time (s): cpu = 00:00:30 ; elapsed = 00:00:41 . Memory (MB): peak = 1834.375 ; gain = 987.477
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
:
%s
*synth2"
 Start Rebuilding User Hierarchy
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2�
�Finished Rebuilding User Hierarchy : Time (s): cpu = 00:00:30 ; elapsed = 00:00:41 . Memory (MB): peak = 1834.375 ; gain = 987.477
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
9
%s
*synth2!
Start Renaming Generated Ports
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2�
�Finished Renaming Generated Ports : Time (s): cpu = 00:00:30 ; elapsed = 00:00:41 . Memory (MB): peak = 1834.375 ; gain = 987.477
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
;
%s
*synth2#
!Start Handling Custom Attributes
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2�
�Finished Handling Custom Attributes : Time (s): cpu = 00:00:30 ; elapsed = 00:00:41 . Memory (MB): peak = 1834.375 ; gain = 987.477
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
8
%s
*synth2 
Start Renaming Generated Nets
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2�
�Finished Renaming Generated Nets : Time (s): cpu = 00:00:30 ; elapsed = 00:00:41 . Memory (MB): peak = 1834.375 ; gain = 987.477
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
9
%s
*synth2!
Start Writing Synthesis Report
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
/
%s
*synth2

Report BlackBoxes: 
h p
x
� 
Q
%s
*synth29
7+------+----------------------------------+----------+
h p
x
� 
Q
%s
*synth29
7|      |BlackBox name                     |Instances |
h p
x
� 
Q
%s
*synth29
7+------+----------------------------------+----------+
h p
x
� 
Q
%s
*synth29
7|1     |matmult_ex_axi_smc_0              |         1|
h p
x
� 
Q
%s
*synth29
7|2     |matmult_ex_matmul_partition_0_0   |         1|
h p
x
� 
Q
%s
*synth29
7|3     |matmult_ex_processing_system7_0_0 |         1|
h p
x
� 
Q
%s
*synth29
7|4     |matmult_ex_rst_ps7_0_100M_0       |         1|
h p
x
� 
Q
%s
*synth29
7+------+----------------------------------+----------+
h p
x
� 
/
%s*synth2

Report Cell Usage: 
h px� 
K
%s*synth23
1+------+--------------------------------+------+
h px� 
K
%s*synth23
1|      |Cell                            |Count |
h px� 
K
%s*synth23
1+------+--------------------------------+------+
h px� 
K
%s*synth23
1|1     |matmult_ex_axi_smc              |     1|
h px� 
K
%s*synth23
1|2     |matmult_ex_matmul_partition_0   |     1|
h px� 
K
%s*synth23
1|3     |matmult_ex_processing_system7_0 |     1|
h px� 
K
%s*synth23
1|4     |matmult_ex_rst_ps7_0_100M       |     1|
h px� 
K
%s*synth23
1+------+--------------------------------+------+
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2�
�Finished Writing Synthesis Report : Time (s): cpu = 00:00:30 ; elapsed = 00:00:41 . Memory (MB): peak = 1834.375 ; gain = 987.477
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
`
%s
*synth2H
FSynthesis finished with 0 errors, 0 critical warnings and 1 warnings.
h p
x
� 
�
%s
*synth2�
Synthesis Optimization Runtime : Time (s): cpu = 00:00:22 ; elapsed = 00:00:38 . Memory (MB): peak = 1834.375 ; gain = 982.188
h p
x
� 
�
%s
*synth2�
�Synthesis Optimization Complete : Time (s): cpu = 00:00:31 ; elapsed = 00:00:41 . Memory (MB): peak = 1834.375 ; gain = 987.477
h p
x
� 
B
 Translating synthesized netlist
350*projectZ1-571h px� 
�
I%sTime (s): cpu = %s ; elapsed = %s . Memory (MB): peak = %s ; gain = %s
268*common2
Netlist sorting complete. 2

00:00:002
00:00:00.0132

1834.3752
0.000Z17-268h px� 
K
)Preparing netlist for logic optimization
349*projectZ1-570h px� 
Q
)Pushed %s inverter(s) to %s load pin(s).
98*opt2
02
0Z31-138h px� 
�
I%sTime (s): cpu = %s ; elapsed = %s . Memory (MB): peak = %s ; gain = %s
268*common2
Netlist sorting complete. 2

00:00:002

00:00:002

1836.3982
0.000Z17-268h px� 
l
!Unisim Transformation Summary:
%s111*project2'
%No Unisim elements were transformed.
Z1-111h px� 
V
%Synth Design complete | Checksum: %s
562*	vivadotcl2

8c8851dbZ4-1430h px� 
C
Releasing license: %s
83*common2
	SynthesisZ17-83h px� 

G%s Infos, %s Warnings, %s Critical Warnings and %s Errors encountered.
28*	vivadotcl2
402
262
02
0Z4-41h px� 
L
%s completed successfully
29*	vivadotcl2
synth_designZ4-42h px� 
�
I%sTime (s): cpu = %s ; elapsed = %s . Memory (MB): peak = %s ; gain = %s
268*common2
synth_design: 2

00:00:352

00:00:512

1840.3752

1224.629Z17-268h px� 
�
 The %s '%s' has been generated.
621*common2

checkpoint2d
bC:/Main_Folder/School/Year_4/Capstone/pynq/pynq_test/pynq_test.runs/synth_1/matmult_ex_wrapper.dcpZ17-1381h px� 
�
Executing command : %s
56330*	planAhead2o
mreport_utilization -file matmult_ex_wrapper_utilization_synth.rpt -pb matmult_ex_wrapper_utilization_synth.pbZ12-24828h px� 
\
Exiting %s at %s...
206*common2
Vivado2
Thu Feb 20 14:42:57 2025Z17-206h px� 


End Record