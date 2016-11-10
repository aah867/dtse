#!/usr/bin/env python

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import step, legend, xlim, ylim, show

from pylab import rcParams
rcParams['figure.figsize'] = 8, 4.9

matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)

INSTR_COUNT = 9
EXEC_TIME = 2
CORE_ENG = 3
CORE_EDP = 4
SPEEDUP = 5
REL_EDP = 6

MAX_N_COL = 100
N_FREQ = 1
N_THREADS = 10
N_KERNL = 3
N_RESULT_SET = 1


freq = (800, 1200, 1600, 2100, 2500, 2900, 3500)		#freq range
thr = ("1", "2", "4", "6", "8", "16", "32", "64", "128", "256", "512")
thr_nr = (1, 2, 4, 6, 8, 16, 32, 64, 128, 256)					#thread
app = ("scalar", "sse_basic", "sse_opt", "sse_cmp", "auto_vec", "avx_512")				#kernel

src = ("l1", "l2", "i", "e")
n_col = (8, 8, 5, 4)
#n_row = (50, 100, 50, 100)
n_row = (50, 50, 50, 50)

base_indx = 0
data = [[[[0 for x in xrange(N_FREQ)] for x in xrange(MAX_N_COL)] for x in xrange(N_THREADS)] for x in xrange(N_KERNL)]

#colors=('darkcyan', 'goldenrod', 'g', 'b', 'r', 'maroon')
colors=('darkcyan', 'g', 'maroon', 'g', 'b', 'r', 'maroon')
markers=('*', '.', 's', '*', '^', '*')
msize=(13, 10, 5, 11, 7, 11)
markColor=('darkcyan', 'goldenrod', 'g', 'b', 'r', 'maroon')
#colors=('coral', 'moccasin', 'ivory')
#markColor=('maroon', 'gold', 'ivory', 'goldenrod', 'darkcyan', 'g')
#colors=('lightgray', 'moccasin', 'ivory', 'goldenrod', 'slateblue')
hatches = ['','//','xx','..','o', '/']
width=0.17

p0={}
p1={}
p2={}
p3={}
p4={}
	
####################
# Detect outliers
####################
def isKeeper( val, mean_val, std_dev ):
	devs = 3
	if ( (val > (mean_val + (devs*std_dev))) or (val < (mean_val - (devs*std_dev))) ):
		deletes = val
		return False
	return True
	
####################
# remove_outliers
####################
def remove_outliers(src, dst, n_row, n_col):

	global base_indx 
	
	e_data = [[[0 for x in xrange(n_row)] for x in xrange(N_FREQ)] for x in xrange(n_col)]
	
	for a in xrange(N_KERNL):
		for t in xrange (N_THREADS):
			f_name = src + "/output_" + app[a] + "_" + thr[t] + ".txt"
			print f_name
			raw_data = [[[0 for x in xrange(n_row)] for x in xrange(N_FREQ)] for x in xrange(n_col)]
			loadFile(f_name, n_row, n_col, raw_data)
			for c in xrange(1, n_col):
				proc_col = [0 for x in xrange(n_col)]
				for f in xrange(N_FREQ):
					mean_val = np.mean(raw_data[c][f])
					std_dev = np.std(raw_data[c][f])
					proc_col = [ d for d in raw_data[c][f] if isKeeper(d, mean_val, std_dev) ]
					dst[a][t][base_indx+c][f] = np.mean(proc_col)
					print dst[a][t][base_indx+c]
	
	base_indx += n_col-1
	print " ****************************************************"
	print "Base indx: %d" %(base_indx) 
	
					
#########################
# Load file
#########################
def loadFile( f_name, n_row, n_col, data_buf ):

	with open(f_name, 'r') as f:
		for i in xrange(N_FREQ):
			line = f.readline()		# skip line
			line = f.readline()		# skip line
			for r in xrange(n_row):
				line = f.readline()
				if not line: break
				values = line.split()
				for c in xrange(1, n_col):
					data_buf[c][i][r] = float(values[c])
	f.close()


#################################
# Exec plot creation function
#################################
def summarize_results(res_set, res_file):

	fd = open(res_file, "w")
	fd.writelines('%15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s\n' % ("#kernel_0", "thread_1", "freq_2", "exec_time_3", "l1_miss_rate_4", "l1_misses_5", "l1_accesses_6", "l1_D_prefetch_7", "l1_D_write_8", "tot_instr_9", "exec_time_10", "l2_miss_rate_11", "l3_miss_rate_12", "l2_misses_13", "l2_accesses_14", "l3_misses_15", "l3_accesses_16", "exec_time_17", "stalled_memsub_18", "stalled_resrc_19", "tot_cyc_20", "exec_time_21", "core_eng_22", "pkg_eng_23", "EDP_core_24", "speedup_25", "EDP_relative_26"))
	
	for a in xrange(N_KERNL):
		for t in xrange(N_THREADS):
			for f in xrange(N_FREQ):
				fd.writelines('%15s %15s %15d ' %(app[a], thr[t], freq[f]))
				for c in xrange(1, base_indx+1):
					fd.writelines('%15.2f ' %(res_set[a][t][c][f]))
				fd.writelines('%15.2f ' %(res_set[a][t][base_indx][f] * res_set[a][t][base_indx-1][f]))
				fd.writelines('%15.2f ' %(res_set[0][0][1][f]/res_set[a][t][1][f]))
				fd.writelines('%15.2f ' %( (res_set[0][0][base_indx][f] * res_set[0][0][base_indx-1][f])/(res_set[a][t][base_indx][f] * res_set[a][t][base_indx-1][f]) ))
				fd.writelines('\n')
	fd.close()


####################
# Draw Line Graphs
####################	
def draw_line_graph(ds_1, ds_2, col_no, plot_name, ylabel):

	data_set = [[[0 for x in xrange(N_FREQ)] for x in xrange(N_KERNL)] for x in xrange(N_THREADS)]
	base_val = [0 for x in xrange(N_FREQ)]

	with open(ds_1, 'r') as fd:
		line = fd.readline()
		for a in xrange(N_KERNL):
			for t in xrange(N_THREADS):
				for f in xrange(N_FREQ):
					line = fd.readline()
					if not line: break
					values = line.split()
					data_set[t][a][f] = float(values[col_no])	
					if (col_no == CORE_EDP):
						data_set[t][a][f] = float(values[3])*float(values[22])
	fd.close()
	
	ind = np.arange(N_THREADS)
	
	for t in xrange(N_THREADS): #N_THREADS
		fig = plt.figure()
		ax1 = fig.add_subplot(2,1,1)
		ax1.set_xticks(ind)
#		ax1.set_xticklabels(['', '800', '1200', '1600','2100','2500','2900','3500'], ha='right')
		
		for a in xrange(N_KERNL):
			p0[a] = ax1.plot(freq, data_set[t][a], '.--', color=colors[a], markersize=msize[a], lw=2.5, dashes=[4,2], marker=markers[a])
			y_start, y_end = ax1.get_ylim()
			if(col_no == EXEC_TIME):
#				ax1.set_ylim(bottom=580, top=4000000000)
				ax1.set_ylim(bottom=7500, top=500000)
			plt.ylabel(ylabel, fontsize=13)
			ax1.set_yscale("log")
			ax1.set_xlabel('Core frequency (MHz)', fontsize=13)
			ax1.set_xlim(580, 3600)
			print data_set[t][a]

		plt_name = ""
		plt_name = plot_name + "_" + str(thr[t]) + ".pdf"
		title = "Real-world dataset " + str(thr[t])
		plt.title(title, fontsize=14)
		l1=ax1.legend( (p0[0][0], p0[1][0], p0[2][0], p0[3][0], p0[4][0]), ("scalar", "sse_basic", "sse_opt", "sse_cmp", "avx512_auto", "avx512_cmp"), loc=1, ncol=3,shadow=False, prop={'size':13} )

#		plt.tight_layout(pad=1, w_pad=0, h_pad=1)

#		plt.savefig(plt_name, format='pdf')
	plt.show()
	
#################################
# Create Bar-chart
#################################

def create_bar_chart(ds_1, ds_2, col_no, plot_name, ylabel):

	data_set_1 = [[[0 for x in xrange(5)] for x in xrange(N_KERNL)] for x in xrange(N_FREQ)]
	data_set_2 = [[[0 for x in xrange(5)] for x in xrange(N_KERNL)] for x in xrange(N_FREQ)]
			
	base_val = [0 for x in xrange(N_FREQ)]
	
	with open(ds_1, 'r') as fd:
#		line = fd.readline()
		for a in xrange(N_KERNL):
			for t in xrange(N_THREADS):
				for f in xrange(N_FREQ): #N_FREQ
					line = fd.readline()
					if not line: break
					values = line.split()					
					print values
					if ( (a==0) and (t==0)):
						if(col_no == SPEEDUP):
							base_val[f] = float(values[2])
						elif(col_no == REL_EDP):
							base_val[f] = float(values[2])*float(values[3])
					if (col_no == SPEEDUP):
						if(t <= 4):
							data_set_1[f][a][t] = base_val[f]/float(values[2])
						else:
							data_set_2[f][a][t-5] = base_val[f]/float(values[2])
					elif (col_no == REL_EDP):
						if(t <= 4):
							data_set_1[f][a][t] = (base_val[f])/(float(values[2])*float(values[3]))
						else:
							data_set_2[f][a][t-5] = (base_val[f])/(float(values[2])*float(values[3]))
						
#						data_set[f][a][t] = float(values[3])
#						print data_set_1[f][a][t]	
					
	fd.close()

	ind = np.arange(5)
	
	for j in xrange(N_FREQ):
	
#		fig = plt.figure(figsize=(8, 4))
		fig = plt.figure()
		ax1 = fig.add_subplot(1,1,1)
		
		for i in xrange(N_KERNL):
			p1[i] = ax1.bar(ind+i*width, data_set_2[j][i], width, color=colors[i], hatch=hatches[i])
			
		for i in xrange(N_THREADS-5):	
			ax1.text(i, data_set_2[j][0][i]+.1, ' %.1f'%data_set_2[j][0][i], va='bottom', fontsize=11, ha='left', rotation=90)	
			ax1.text(i+1*width, data_set_2[j][1][i]+.1, ' %.1f'%data_set_2[j][1][i], va='bottom', fontsize=11, ha='left', rotation=90)	
			ax1.text(i+2*width, data_set_2[j][2][i]+.1, ' %.1f'%data_set_2[j][2][i], va='bottom', fontsize=11, rotation=90)	
#			ax1.text(i+3*width, data_set_2[j][3][i]+.1, ' %.1f'%data_set_2[j][3][i], va='bottom', fontsize=9.5, rotation=90)	
#			ax1.text(i+4*width, data_set_2[j][4][i]+.1, ' %.1f'%data_set_2[j][4][i], va='bottom', fontsize=9.5, rotation=90)	
#			ax1.text(i+5*width, data_set_2[j][5][i]+.1, ' %.1f'%data_set_2[j][5][i], va='bottom', fontsize=9.5, rotation=90)	

		ax1.set_xticks(ind+i*width)
#		ax1.set_xticklabels(['1', '2', '4', '6', '8', '16', '32', '64', '128', '256'], ha='right')
		ax1.set_xticklabels(['16', '32', '64', '128', '256','0'], ha='center')

		y_start, y_end = ax1.get_ylim()
		ax1.set_xlim(0, 4.8)
		if (col_no == SPEEDUP):
			ax1.set_ylim(bottom=0, top=250)	#without log
#			ax1.set_yscale("log")
#			ax1.set_ylim(bottom=14, top=2500)	#with log
		elif (col_no == REL_EDP):
			ax1.set_yscale("log")
			ax1.set_ylim(bottom=1, top=95000000)
			
				
		ax1.set_ylabel(ylabel, fontsize=15)
		ax1.set_xlabel('No of threads', fontsize=15)
#		ax1.set_title('Real-world dataset', fontsize=13)
		l1=ax1.legend( (p1[0], p1[1], p1[2]), ("scalar_E0", "scalar_E2", "AVX-512_E2"), loc=2, ncol=3,shadow=False, prop={'size':14} )
		plt.tight_layout(pad=0.1, w_pad=0, h_pad=0.2)
#		plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

		plt_name = ""
		plt_name = plot_name + "_256.pdf"
		plt.tight_layout(pad=1, w_pad=0, h_pad=1)
		plt.savefig(plt_name, format='pdf')	
	
#	plt.show()

	
def mains():

	ds_1 = "dtse_knl.txt"
	res = [[[[0 for x in xrange(N_FREQ)] for x in xrange(MAX_N_COL)] for x in xrange(N_THREADS)] for x in xrange(N_KERNL)]
	
#	draw_line_graph(ds_1, ds_1, EXEC_TIME, "exec_time", "Execution time ($\mu s$)")
	create_bar_chart(ds_1, ds_1,SPEEDUP, "knl_speedup", "Speedup")
#	draw_line_graph(ds_1, ds_2, CORE_ENG, "core_eng", "Core energy consumption ($mj$)")
#	draw_line_graph(ds_1, ds_2, CORE_EDP, "core_edp", "Core energy efficiency (EDP)")
#	create_bar_chart(ds_1, ds_1, REL_EDP, "knl_rel_edp", "Relative core energy efficiency (EDP)")
	#draw_line_graph(ds_1, ds_2, INSTR_COUNT, "instruction", "Total number of instructions")

	
mains()
