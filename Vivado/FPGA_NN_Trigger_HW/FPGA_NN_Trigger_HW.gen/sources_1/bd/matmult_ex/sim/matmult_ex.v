//Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
//Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
//--------------------------------------------------------------------------------
//Tool Version: Vivado v.2024.2 (win64) Build 5239630 Fri Nov 08 22:35:27 MST 2024
//Date        : Thu Feb 20 14:40:33 2025
//Host        : TABLET-60QL2L1F running 64-bit major release  (build 9200)
//Command     : generate_target matmult_ex.bd
//Design      : matmult_ex
//Purpose     : IP block netlist
//--------------------------------------------------------------------------------
`timescale 1 ps / 1 ps

(* CORE_GENERATION_INFO = "matmult_ex,IP_Integrator,{x_ipVendor=xilinx.com,x_ipLibrary=BlockDiagram,x_ipName=matmult_ex,x_ipVersion=1.00.a,x_ipLanguage=VERILOG,numBlks=6,numReposBlks=6,numNonXlnxBlks=0,numHierBlks=0,maxHierDepth=0,numSysgenBlks=0,numHlsBlks=1,numHdlrefBlks=0,numPkgbdBlks=0,bdsource=USER,da_axi4_cnt=2,da_ps7_cnt=1,synth_mode=Hierarchical}" *) (* HW_HANDOFF = "matmult_ex.hwdef" *) 
module matmult_ex
   (DDR_addr,
    DDR_ba,
    DDR_cas_n,
    DDR_ck_n,
    DDR_ck_p,
    DDR_cke,
    DDR_cs_n,
    DDR_dm,
    DDR_dq,
    DDR_dqs_n,
    DDR_dqs_p,
    DDR_odt,
    DDR_ras_n,
    DDR_reset_n,
    DDR_we_n,
    FIXED_IO_ddr_vrn,
    FIXED_IO_ddr_vrp,
    FIXED_IO_mio,
    FIXED_IO_ps_clk,
    FIXED_IO_ps_porb,
    FIXED_IO_ps_srstb);
  (* X_INTERFACE_INFO = "xilinx.com:interface:ddrx:1.0 DDR ADDR" *) (* X_INTERFACE_MODE = "Master" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DDR, AXI_ARBITRATION_SCHEME TDM, BURST_LENGTH 8, CAN_DEBUG false, CAS_LATENCY 11, CAS_WRITE_LATENCY 11, CS_ENABLED true, DATA_MASK_ENABLED true, DATA_WIDTH 8, MEMORY_TYPE COMPONENTS, MEM_ADDR_MAP ROW_COLUMN_BANK, SLOT Single, TIMEPERIOD_PS 1250" *) inout [14:0]DDR_addr;
  (* X_INTERFACE_INFO = "xilinx.com:interface:ddrx:1.0 DDR BA" *) inout [2:0]DDR_ba;
  (* X_INTERFACE_INFO = "xilinx.com:interface:ddrx:1.0 DDR CAS_N" *) inout DDR_cas_n;
  (* X_INTERFACE_INFO = "xilinx.com:interface:ddrx:1.0 DDR CK_N" *) inout DDR_ck_n;
  (* X_INTERFACE_INFO = "xilinx.com:interface:ddrx:1.0 DDR CK_P" *) inout DDR_ck_p;
  (* X_INTERFACE_INFO = "xilinx.com:interface:ddrx:1.0 DDR CKE" *) inout DDR_cke;
  (* X_INTERFACE_INFO = "xilinx.com:interface:ddrx:1.0 DDR CS_N" *) inout DDR_cs_n;
  (* X_INTERFACE_INFO = "xilinx.com:interface:ddrx:1.0 DDR DM" *) inout [3:0]DDR_dm;
  (* X_INTERFACE_INFO = "xilinx.com:interface:ddrx:1.0 DDR DQ" *) inout [31:0]DDR_dq;
  (* X_INTERFACE_INFO = "xilinx.com:interface:ddrx:1.0 DDR DQS_N" *) inout [3:0]DDR_dqs_n;
  (* X_INTERFACE_INFO = "xilinx.com:interface:ddrx:1.0 DDR DQS_P" *) inout [3:0]DDR_dqs_p;
  (* X_INTERFACE_INFO = "xilinx.com:interface:ddrx:1.0 DDR ODT" *) inout DDR_odt;
  (* X_INTERFACE_INFO = "xilinx.com:interface:ddrx:1.0 DDR RAS_N" *) inout DDR_ras_n;
  (* X_INTERFACE_INFO = "xilinx.com:interface:ddrx:1.0 DDR RESET_N" *) inout DDR_reset_n;
  (* X_INTERFACE_INFO = "xilinx.com:interface:ddrx:1.0 DDR WE_N" *) inout DDR_we_n;
  (* X_INTERFACE_INFO = "xilinx.com:display_processing_system7:fixedio:1.0 FIXED_IO DDR_VRN" *) (* X_INTERFACE_MODE = "Master" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME FIXED_IO, CAN_DEBUG false" *) inout FIXED_IO_ddr_vrn;
  (* X_INTERFACE_INFO = "xilinx.com:display_processing_system7:fixedio:1.0 FIXED_IO DDR_VRP" *) inout FIXED_IO_ddr_vrp;
  (* X_INTERFACE_INFO = "xilinx.com:display_processing_system7:fixedio:1.0 FIXED_IO MIO" *) inout [53:0]FIXED_IO_mio;
  (* X_INTERFACE_INFO = "xilinx.com:display_processing_system7:fixedio:1.0 FIXED_IO PS_CLK" *) inout FIXED_IO_ps_clk;
  (* X_INTERFACE_INFO = "xilinx.com:display_processing_system7:fixedio:1.0 FIXED_IO PS_PORB" *) inout FIXED_IO_ps_porb;
  (* X_INTERFACE_INFO = "xilinx.com:display_processing_system7:fixedio:1.0 FIXED_IO PS_SRSTB" *) inout FIXED_IO_ps_srstb;

  wire [14:0]DDR_addr;
  wire [2:0]DDR_ba;
  wire DDR_cas_n;
  wire DDR_ck_n;
  wire DDR_ck_p;
  wire DDR_cke;
  wire DDR_cs_n;
  wire [3:0]DDR_dm;
  wire [31:0]DDR_dq;
  wire [3:0]DDR_dqs_n;
  wire [3:0]DDR_dqs_p;
  wire DDR_odt;
  wire DDR_ras_n;
  wire DDR_reset_n;
  wire DDR_we_n;
  wire FIXED_IO_ddr_vrn;
  wire FIXED_IO_ddr_vrp;
  wire [53:0]FIXED_IO_mio;
  wire FIXED_IO_ps_clk;
  wire FIXED_IO_ps_porb;
  wire FIXED_IO_ps_srstb;
  wire [5:0]axi_smc_M00_AXI_ARADDR;
  wire axi_smc_M00_AXI_ARREADY;
  wire axi_smc_M00_AXI_ARVALID;
  wire [5:0]axi_smc_M00_AXI_AWADDR;
  wire axi_smc_M00_AXI_AWREADY;
  wire axi_smc_M00_AXI_AWVALID;
  wire axi_smc_M00_AXI_BREADY;
  wire [1:0]axi_smc_M00_AXI_BRESP;
  wire axi_smc_M00_AXI_BVALID;
  wire [31:0]axi_smc_M00_AXI_RDATA;
  wire axi_smc_M00_AXI_RREADY;
  wire [1:0]axi_smc_M00_AXI_RRESP;
  wire axi_smc_M00_AXI_RVALID;
  wire [31:0]axi_smc_M00_AXI_WDATA;
  wire axi_smc_M00_AXI_WREADY;
  wire [3:0]axi_smc_M00_AXI_WSTRB;
  wire axi_smc_M00_AXI_WVALID;
  wire [63:0]matmul_partition_0_m_axi_gmem_ARADDR;
  wire [1:0]matmul_partition_0_m_axi_gmem_ARBURST;
  wire [3:0]matmul_partition_0_m_axi_gmem_ARCACHE;
  wire [0:0]matmul_partition_0_m_axi_gmem_ARID;
  wire [7:0]matmul_partition_0_m_axi_gmem_ARLEN;
  wire [1:0]matmul_partition_0_m_axi_gmem_ARLOCK;
  wire [2:0]matmul_partition_0_m_axi_gmem_ARPROT;
  wire [3:0]matmul_partition_0_m_axi_gmem_ARQOS;
  wire matmul_partition_0_m_axi_gmem_ARREADY;
  wire [2:0]matmul_partition_0_m_axi_gmem_ARSIZE;
  wire matmul_partition_0_m_axi_gmem_ARVALID;
  wire [63:0]matmul_partition_0_m_axi_gmem_AWADDR;
  wire [1:0]matmul_partition_0_m_axi_gmem_AWBURST;
  wire [3:0]matmul_partition_0_m_axi_gmem_AWCACHE;
  wire [0:0]matmul_partition_0_m_axi_gmem_AWID;
  wire [7:0]matmul_partition_0_m_axi_gmem_AWLEN;
  wire [1:0]matmul_partition_0_m_axi_gmem_AWLOCK;
  wire [2:0]matmul_partition_0_m_axi_gmem_AWPROT;
  wire [3:0]matmul_partition_0_m_axi_gmem_AWQOS;
  wire matmul_partition_0_m_axi_gmem_AWREADY;
  wire [2:0]matmul_partition_0_m_axi_gmem_AWSIZE;
  wire matmul_partition_0_m_axi_gmem_AWVALID;
  wire [0:0]matmul_partition_0_m_axi_gmem_BID;
  wire matmul_partition_0_m_axi_gmem_BREADY;
  wire [1:0]matmul_partition_0_m_axi_gmem_BRESP;
  wire matmul_partition_0_m_axi_gmem_BVALID;
  wire [31:0]matmul_partition_0_m_axi_gmem_RDATA;
  wire [0:0]matmul_partition_0_m_axi_gmem_RID;
  wire matmul_partition_0_m_axi_gmem_RLAST;
  wire matmul_partition_0_m_axi_gmem_RREADY;
  wire [1:0]matmul_partition_0_m_axi_gmem_RRESP;
  wire matmul_partition_0_m_axi_gmem_RVALID;
  wire [31:0]matmul_partition_0_m_axi_gmem_WDATA;
  wire matmul_partition_0_m_axi_gmem_WLAST;
  wire matmul_partition_0_m_axi_gmem_WREADY;
  wire [3:0]matmul_partition_0_m_axi_gmem_WSTRB;
  wire matmul_partition_0_m_axi_gmem_WVALID;
  wire processing_system7_0_FCLK_CLK0;
  wire processing_system7_0_FCLK_RESET0_N;
  wire [31:0]processing_system7_0_M_AXI_GP0_ARADDR;
  wire [1:0]processing_system7_0_M_AXI_GP0_ARBURST;
  wire [3:0]processing_system7_0_M_AXI_GP0_ARCACHE;
  wire [11:0]processing_system7_0_M_AXI_GP0_ARID;
  wire [3:0]processing_system7_0_M_AXI_GP0_ARLEN;
  wire [1:0]processing_system7_0_M_AXI_GP0_ARLOCK;
  wire [2:0]processing_system7_0_M_AXI_GP0_ARPROT;
  wire [3:0]processing_system7_0_M_AXI_GP0_ARQOS;
  wire processing_system7_0_M_AXI_GP0_ARREADY;
  wire [2:0]processing_system7_0_M_AXI_GP0_ARSIZE;
  wire processing_system7_0_M_AXI_GP0_ARVALID;
  wire [31:0]processing_system7_0_M_AXI_GP0_AWADDR;
  wire [1:0]processing_system7_0_M_AXI_GP0_AWBURST;
  wire [3:0]processing_system7_0_M_AXI_GP0_AWCACHE;
  wire [11:0]processing_system7_0_M_AXI_GP0_AWID;
  wire [3:0]processing_system7_0_M_AXI_GP0_AWLEN;
  wire [1:0]processing_system7_0_M_AXI_GP0_AWLOCK;
  wire [2:0]processing_system7_0_M_AXI_GP0_AWPROT;
  wire [3:0]processing_system7_0_M_AXI_GP0_AWQOS;
  wire processing_system7_0_M_AXI_GP0_AWREADY;
  wire [2:0]processing_system7_0_M_AXI_GP0_AWSIZE;
  wire processing_system7_0_M_AXI_GP0_AWVALID;
  wire [11:0]processing_system7_0_M_AXI_GP0_BID;
  wire processing_system7_0_M_AXI_GP0_BREADY;
  wire [1:0]processing_system7_0_M_AXI_GP0_BRESP;
  wire processing_system7_0_M_AXI_GP0_BVALID;
  wire [31:0]processing_system7_0_M_AXI_GP0_RDATA;
  wire [11:0]processing_system7_0_M_AXI_GP0_RID;
  wire processing_system7_0_M_AXI_GP0_RLAST;
  wire processing_system7_0_M_AXI_GP0_RREADY;
  wire [1:0]processing_system7_0_M_AXI_GP0_RRESP;
  wire processing_system7_0_M_AXI_GP0_RVALID;
  wire [31:0]processing_system7_0_M_AXI_GP0_WDATA;
  wire [11:0]processing_system7_0_M_AXI_GP0_WID;
  wire processing_system7_0_M_AXI_GP0_WLAST;
  wire processing_system7_0_M_AXI_GP0_WREADY;
  wire [3:0]processing_system7_0_M_AXI_GP0_WSTRB;
  wire processing_system7_0_M_AXI_GP0_WVALID;
  wire [0:0]rst_ps7_0_100M_peripheral_aresetn;
  wire [31:0]xlconstant_0_dout;
  wire [31:0]xlconstant_1_dout;

  matmult_ex_axi_smc_0 axi_smc
       (.M00_AXI_araddr(axi_smc_M00_AXI_ARADDR),
        .M00_AXI_arready(axi_smc_M00_AXI_ARREADY),
        .M00_AXI_arvalid(axi_smc_M00_AXI_ARVALID),
        .M00_AXI_awaddr(axi_smc_M00_AXI_AWADDR),
        .M00_AXI_awready(axi_smc_M00_AXI_AWREADY),
        .M00_AXI_awvalid(axi_smc_M00_AXI_AWVALID),
        .M00_AXI_bready(axi_smc_M00_AXI_BREADY),
        .M00_AXI_bresp(axi_smc_M00_AXI_BRESP),
        .M00_AXI_bvalid(axi_smc_M00_AXI_BVALID),
        .M00_AXI_rdata(axi_smc_M00_AXI_RDATA),
        .M00_AXI_rready(axi_smc_M00_AXI_RREADY),
        .M00_AXI_rresp(axi_smc_M00_AXI_RRESP),
        .M00_AXI_rvalid(axi_smc_M00_AXI_RVALID),
        .M00_AXI_wdata(axi_smc_M00_AXI_WDATA),
        .M00_AXI_wready(axi_smc_M00_AXI_WREADY),
        .M00_AXI_wstrb(axi_smc_M00_AXI_WSTRB),
        .M00_AXI_wvalid(axi_smc_M00_AXI_WVALID),
        .S00_AXI_araddr(processing_system7_0_M_AXI_GP0_ARADDR),
        .S00_AXI_arburst(processing_system7_0_M_AXI_GP0_ARBURST),
        .S00_AXI_arcache(processing_system7_0_M_AXI_GP0_ARCACHE),
        .S00_AXI_arid(processing_system7_0_M_AXI_GP0_ARID),
        .S00_AXI_arlen(processing_system7_0_M_AXI_GP0_ARLEN),
        .S00_AXI_arlock(processing_system7_0_M_AXI_GP0_ARLOCK),
        .S00_AXI_arprot(processing_system7_0_M_AXI_GP0_ARPROT),
        .S00_AXI_arqos(processing_system7_0_M_AXI_GP0_ARQOS),
        .S00_AXI_arready(processing_system7_0_M_AXI_GP0_ARREADY),
        .S00_AXI_arsize(processing_system7_0_M_AXI_GP0_ARSIZE),
        .S00_AXI_arvalid(processing_system7_0_M_AXI_GP0_ARVALID),
        .S00_AXI_awaddr(processing_system7_0_M_AXI_GP0_AWADDR),
        .S00_AXI_awburst(processing_system7_0_M_AXI_GP0_AWBURST),
        .S00_AXI_awcache(processing_system7_0_M_AXI_GP0_AWCACHE),
        .S00_AXI_awid(processing_system7_0_M_AXI_GP0_AWID),
        .S00_AXI_awlen(processing_system7_0_M_AXI_GP0_AWLEN),
        .S00_AXI_awlock(processing_system7_0_M_AXI_GP0_AWLOCK),
        .S00_AXI_awprot(processing_system7_0_M_AXI_GP0_AWPROT),
        .S00_AXI_awqos(processing_system7_0_M_AXI_GP0_AWQOS),
        .S00_AXI_awready(processing_system7_0_M_AXI_GP0_AWREADY),
        .S00_AXI_awsize(processing_system7_0_M_AXI_GP0_AWSIZE),
        .S00_AXI_awvalid(processing_system7_0_M_AXI_GP0_AWVALID),
        .S00_AXI_bid(processing_system7_0_M_AXI_GP0_BID),
        .S00_AXI_bready(processing_system7_0_M_AXI_GP0_BREADY),
        .S00_AXI_bresp(processing_system7_0_M_AXI_GP0_BRESP),
        .S00_AXI_bvalid(processing_system7_0_M_AXI_GP0_BVALID),
        .S00_AXI_rdata(processing_system7_0_M_AXI_GP0_RDATA),
        .S00_AXI_rid(processing_system7_0_M_AXI_GP0_RID),
        .S00_AXI_rlast(processing_system7_0_M_AXI_GP0_RLAST),
        .S00_AXI_rready(processing_system7_0_M_AXI_GP0_RREADY),
        .S00_AXI_rresp(processing_system7_0_M_AXI_GP0_RRESP),
        .S00_AXI_rvalid(processing_system7_0_M_AXI_GP0_RVALID),
        .S00_AXI_wdata(processing_system7_0_M_AXI_GP0_WDATA),
        .S00_AXI_wid(processing_system7_0_M_AXI_GP0_WID),
        .S00_AXI_wlast(processing_system7_0_M_AXI_GP0_WLAST),
        .S00_AXI_wready(processing_system7_0_M_AXI_GP0_WREADY),
        .S00_AXI_wstrb(processing_system7_0_M_AXI_GP0_WSTRB),
        .S00_AXI_wvalid(processing_system7_0_M_AXI_GP0_WVALID),
        .S01_AXI_araddr(matmul_partition_0_m_axi_gmem_ARADDR),
        .S01_AXI_arburst(matmul_partition_0_m_axi_gmem_ARBURST),
        .S01_AXI_arcache(matmul_partition_0_m_axi_gmem_ARCACHE),
        .S01_AXI_arid(matmul_partition_0_m_axi_gmem_ARID),
        .S01_AXI_arlen(matmul_partition_0_m_axi_gmem_ARLEN),
        .S01_AXI_arlock(matmul_partition_0_m_axi_gmem_ARLOCK[0]),
        .S01_AXI_arprot(matmul_partition_0_m_axi_gmem_ARPROT),
        .S01_AXI_arqos(matmul_partition_0_m_axi_gmem_ARQOS),
        .S01_AXI_arready(matmul_partition_0_m_axi_gmem_ARREADY),
        .S01_AXI_arsize(matmul_partition_0_m_axi_gmem_ARSIZE),
        .S01_AXI_arvalid(matmul_partition_0_m_axi_gmem_ARVALID),
        .S01_AXI_awaddr(matmul_partition_0_m_axi_gmem_AWADDR),
        .S01_AXI_awburst(matmul_partition_0_m_axi_gmem_AWBURST),
        .S01_AXI_awcache(matmul_partition_0_m_axi_gmem_AWCACHE),
        .S01_AXI_awid(matmul_partition_0_m_axi_gmem_AWID),
        .S01_AXI_awlen(matmul_partition_0_m_axi_gmem_AWLEN),
        .S01_AXI_awlock(matmul_partition_0_m_axi_gmem_AWLOCK[0]),
        .S01_AXI_awprot(matmul_partition_0_m_axi_gmem_AWPROT),
        .S01_AXI_awqos(matmul_partition_0_m_axi_gmem_AWQOS),
        .S01_AXI_awready(matmul_partition_0_m_axi_gmem_AWREADY),
        .S01_AXI_awsize(matmul_partition_0_m_axi_gmem_AWSIZE),
        .S01_AXI_awvalid(matmul_partition_0_m_axi_gmem_AWVALID),
        .S01_AXI_bid(matmul_partition_0_m_axi_gmem_BID),
        .S01_AXI_bready(matmul_partition_0_m_axi_gmem_BREADY),
        .S01_AXI_bresp(matmul_partition_0_m_axi_gmem_BRESP),
        .S01_AXI_bvalid(matmul_partition_0_m_axi_gmem_BVALID),
        .S01_AXI_rdata(matmul_partition_0_m_axi_gmem_RDATA),
        .S01_AXI_rid(matmul_partition_0_m_axi_gmem_RID),
        .S01_AXI_rlast(matmul_partition_0_m_axi_gmem_RLAST),
        .S01_AXI_rready(matmul_partition_0_m_axi_gmem_RREADY),
        .S01_AXI_rresp(matmul_partition_0_m_axi_gmem_RRESP),
        .S01_AXI_rvalid(matmul_partition_0_m_axi_gmem_RVALID),
        .S01_AXI_wdata(matmul_partition_0_m_axi_gmem_WDATA),
        .S01_AXI_wlast(matmul_partition_0_m_axi_gmem_WLAST),
        .S01_AXI_wready(matmul_partition_0_m_axi_gmem_WREADY),
        .S01_AXI_wstrb(matmul_partition_0_m_axi_gmem_WSTRB),
        .S01_AXI_wvalid(matmul_partition_0_m_axi_gmem_WVALID),
        .aclk(processing_system7_0_FCLK_CLK0),
        .aresetn(rst_ps7_0_100M_peripheral_aresetn));
  matmult_ex_matmul_partition_0_0 matmul_partition_0
       (.ap_clk(processing_system7_0_FCLK_CLK0),
        .ap_rst_n(rst_ps7_0_100M_peripheral_aresetn),
        .dim(xlconstant_0_dout),
        .m_axi_gmem_ARADDR(matmul_partition_0_m_axi_gmem_ARADDR),
        .m_axi_gmem_ARBURST(matmul_partition_0_m_axi_gmem_ARBURST),
        .m_axi_gmem_ARCACHE(matmul_partition_0_m_axi_gmem_ARCACHE),
        .m_axi_gmem_ARID(matmul_partition_0_m_axi_gmem_ARID),
        .m_axi_gmem_ARLEN(matmul_partition_0_m_axi_gmem_ARLEN),
        .m_axi_gmem_ARLOCK(matmul_partition_0_m_axi_gmem_ARLOCK),
        .m_axi_gmem_ARPROT(matmul_partition_0_m_axi_gmem_ARPROT),
        .m_axi_gmem_ARQOS(matmul_partition_0_m_axi_gmem_ARQOS),
        .m_axi_gmem_ARREADY(matmul_partition_0_m_axi_gmem_ARREADY),
        .m_axi_gmem_ARSIZE(matmul_partition_0_m_axi_gmem_ARSIZE),
        .m_axi_gmem_ARVALID(matmul_partition_0_m_axi_gmem_ARVALID),
        .m_axi_gmem_AWADDR(matmul_partition_0_m_axi_gmem_AWADDR),
        .m_axi_gmem_AWBURST(matmul_partition_0_m_axi_gmem_AWBURST),
        .m_axi_gmem_AWCACHE(matmul_partition_0_m_axi_gmem_AWCACHE),
        .m_axi_gmem_AWID(matmul_partition_0_m_axi_gmem_AWID),
        .m_axi_gmem_AWLEN(matmul_partition_0_m_axi_gmem_AWLEN),
        .m_axi_gmem_AWLOCK(matmul_partition_0_m_axi_gmem_AWLOCK),
        .m_axi_gmem_AWPROT(matmul_partition_0_m_axi_gmem_AWPROT),
        .m_axi_gmem_AWQOS(matmul_partition_0_m_axi_gmem_AWQOS),
        .m_axi_gmem_AWREADY(matmul_partition_0_m_axi_gmem_AWREADY),
        .m_axi_gmem_AWSIZE(matmul_partition_0_m_axi_gmem_AWSIZE),
        .m_axi_gmem_AWVALID(matmul_partition_0_m_axi_gmem_AWVALID),
        .m_axi_gmem_BID(matmul_partition_0_m_axi_gmem_BID),
        .m_axi_gmem_BREADY(matmul_partition_0_m_axi_gmem_BREADY),
        .m_axi_gmem_BRESP(matmul_partition_0_m_axi_gmem_BRESP),
        .m_axi_gmem_BVALID(matmul_partition_0_m_axi_gmem_BVALID),
        .m_axi_gmem_RDATA(matmul_partition_0_m_axi_gmem_RDATA),
        .m_axi_gmem_RID(matmul_partition_0_m_axi_gmem_RID),
        .m_axi_gmem_RLAST(matmul_partition_0_m_axi_gmem_RLAST),
        .m_axi_gmem_RREADY(matmul_partition_0_m_axi_gmem_RREADY),
        .m_axi_gmem_RRESP(matmul_partition_0_m_axi_gmem_RRESP),
        .m_axi_gmem_RVALID(matmul_partition_0_m_axi_gmem_RVALID),
        .m_axi_gmem_WDATA(matmul_partition_0_m_axi_gmem_WDATA),
        .m_axi_gmem_WLAST(matmul_partition_0_m_axi_gmem_WLAST),
        .m_axi_gmem_WREADY(matmul_partition_0_m_axi_gmem_WREADY),
        .m_axi_gmem_WSTRB(matmul_partition_0_m_axi_gmem_WSTRB),
        .m_axi_gmem_WVALID(matmul_partition_0_m_axi_gmem_WVALID),
        .rep_count(xlconstant_1_dout),
        .s_axi_control_ARADDR(axi_smc_M00_AXI_ARADDR),
        .s_axi_control_ARREADY(axi_smc_M00_AXI_ARREADY),
        .s_axi_control_ARVALID(axi_smc_M00_AXI_ARVALID),
        .s_axi_control_AWADDR(axi_smc_M00_AXI_AWADDR),
        .s_axi_control_AWREADY(axi_smc_M00_AXI_AWREADY),
        .s_axi_control_AWVALID(axi_smc_M00_AXI_AWVALID),
        .s_axi_control_BREADY(axi_smc_M00_AXI_BREADY),
        .s_axi_control_BRESP(axi_smc_M00_AXI_BRESP),
        .s_axi_control_BVALID(axi_smc_M00_AXI_BVALID),
        .s_axi_control_RDATA(axi_smc_M00_AXI_RDATA),
        .s_axi_control_RREADY(axi_smc_M00_AXI_RREADY),
        .s_axi_control_RRESP(axi_smc_M00_AXI_RRESP),
        .s_axi_control_RVALID(axi_smc_M00_AXI_RVALID),
        .s_axi_control_WDATA(axi_smc_M00_AXI_WDATA),
        .s_axi_control_WREADY(axi_smc_M00_AXI_WREADY),
        .s_axi_control_WSTRB(axi_smc_M00_AXI_WSTRB),
        .s_axi_control_WVALID(axi_smc_M00_AXI_WVALID));
  matmult_ex_processing_system7_0_0 processing_system7_0
       (.DDR_Addr(DDR_addr),
        .DDR_BankAddr(DDR_ba),
        .DDR_CAS_n(DDR_cas_n),
        .DDR_CKE(DDR_cke),
        .DDR_CS_n(DDR_cs_n),
        .DDR_Clk(DDR_ck_p),
        .DDR_Clk_n(DDR_ck_n),
        .DDR_DM(DDR_dm),
        .DDR_DQ(DDR_dq),
        .DDR_DQS(DDR_dqs_p),
        .DDR_DQS_n(DDR_dqs_n),
        .DDR_DRSTB(DDR_reset_n),
        .DDR_ODT(DDR_odt),
        .DDR_RAS_n(DDR_ras_n),
        .DDR_VRN(FIXED_IO_ddr_vrn),
        .DDR_VRP(FIXED_IO_ddr_vrp),
        .DDR_WEB(DDR_we_n),
        .FCLK_CLK0(processing_system7_0_FCLK_CLK0),
        .FCLK_RESET0_N(processing_system7_0_FCLK_RESET0_N),
        .MIO(FIXED_IO_mio),
        .M_AXI_GP0_ACLK(processing_system7_0_FCLK_CLK0),
        .M_AXI_GP0_ARADDR(processing_system7_0_M_AXI_GP0_ARADDR),
        .M_AXI_GP0_ARBURST(processing_system7_0_M_AXI_GP0_ARBURST),
        .M_AXI_GP0_ARCACHE(processing_system7_0_M_AXI_GP0_ARCACHE),
        .M_AXI_GP0_ARID(processing_system7_0_M_AXI_GP0_ARID),
        .M_AXI_GP0_ARLEN(processing_system7_0_M_AXI_GP0_ARLEN),
        .M_AXI_GP0_ARLOCK(processing_system7_0_M_AXI_GP0_ARLOCK),
        .M_AXI_GP0_ARPROT(processing_system7_0_M_AXI_GP0_ARPROT),
        .M_AXI_GP0_ARQOS(processing_system7_0_M_AXI_GP0_ARQOS),
        .M_AXI_GP0_ARREADY(processing_system7_0_M_AXI_GP0_ARREADY),
        .M_AXI_GP0_ARSIZE(processing_system7_0_M_AXI_GP0_ARSIZE),
        .M_AXI_GP0_ARVALID(processing_system7_0_M_AXI_GP0_ARVALID),
        .M_AXI_GP0_AWADDR(processing_system7_0_M_AXI_GP0_AWADDR),
        .M_AXI_GP0_AWBURST(processing_system7_0_M_AXI_GP0_AWBURST),
        .M_AXI_GP0_AWCACHE(processing_system7_0_M_AXI_GP0_AWCACHE),
        .M_AXI_GP0_AWID(processing_system7_0_M_AXI_GP0_AWID),
        .M_AXI_GP0_AWLEN(processing_system7_0_M_AXI_GP0_AWLEN),
        .M_AXI_GP0_AWLOCK(processing_system7_0_M_AXI_GP0_AWLOCK),
        .M_AXI_GP0_AWPROT(processing_system7_0_M_AXI_GP0_AWPROT),
        .M_AXI_GP0_AWQOS(processing_system7_0_M_AXI_GP0_AWQOS),
        .M_AXI_GP0_AWREADY(processing_system7_0_M_AXI_GP0_AWREADY),
        .M_AXI_GP0_AWSIZE(processing_system7_0_M_AXI_GP0_AWSIZE),
        .M_AXI_GP0_AWVALID(processing_system7_0_M_AXI_GP0_AWVALID),
        .M_AXI_GP0_BID(processing_system7_0_M_AXI_GP0_BID),
        .M_AXI_GP0_BREADY(processing_system7_0_M_AXI_GP0_BREADY),
        .M_AXI_GP0_BRESP(processing_system7_0_M_AXI_GP0_BRESP),
        .M_AXI_GP0_BVALID(processing_system7_0_M_AXI_GP0_BVALID),
        .M_AXI_GP0_RDATA(processing_system7_0_M_AXI_GP0_RDATA),
        .M_AXI_GP0_RID(processing_system7_0_M_AXI_GP0_RID),
        .M_AXI_GP0_RLAST(processing_system7_0_M_AXI_GP0_RLAST),
        .M_AXI_GP0_RREADY(processing_system7_0_M_AXI_GP0_RREADY),
        .M_AXI_GP0_RRESP(processing_system7_0_M_AXI_GP0_RRESP),
        .M_AXI_GP0_RVALID(processing_system7_0_M_AXI_GP0_RVALID),
        .M_AXI_GP0_WDATA(processing_system7_0_M_AXI_GP0_WDATA),
        .M_AXI_GP0_WID(processing_system7_0_M_AXI_GP0_WID),
        .M_AXI_GP0_WLAST(processing_system7_0_M_AXI_GP0_WLAST),
        .M_AXI_GP0_WREADY(processing_system7_0_M_AXI_GP0_WREADY),
        .M_AXI_GP0_WSTRB(processing_system7_0_M_AXI_GP0_WSTRB),
        .M_AXI_GP0_WVALID(processing_system7_0_M_AXI_GP0_WVALID),
        .PS_CLK(FIXED_IO_ps_clk),
        .PS_PORB(FIXED_IO_ps_porb),
        .PS_SRSTB(FIXED_IO_ps_srstb),
        .USB0_VBUS_PWRFAULT(1'b0));
  matmult_ex_rst_ps7_0_100M_0 rst_ps7_0_100M
       (.aux_reset_in(1'b1),
        .dcm_locked(1'b1),
        .ext_reset_in(processing_system7_0_FCLK_RESET0_N),
        .mb_debug_sys_rst(1'b0),
        .peripheral_aresetn(rst_ps7_0_100M_peripheral_aresetn),
        .slowest_sync_clk(processing_system7_0_FCLK_CLK0));
  matmult_ex_xlconstant_0_0 xlconstant_0
       (.dout(xlconstant_0_dout));
  matmult_ex_xlconstant_0_1 xlconstant_1
       (.dout(xlconstant_1_dout));
endmodule
