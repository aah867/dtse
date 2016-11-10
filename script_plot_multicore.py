#!/usr/bin/env python

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import step, legend, xlim, ylim, show

matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)

INSTR_COUNT = 9
EXEC_TIME = 3
CORE_ENG = 22
CORE_EDP = 24
SPEEDUP = 25
REL_EDP = 26
MEM_STALL = 30

MAX_N_COL = 100
N_FREQ = 7
N_THREADS = 5
N_KERNL = 12
N_RESULT_SET = 4
N_ROWS = 50

freq = (800, 1200, 1600, 2100, 2500, 2900, 3500)		#freq range
thr = ("1", "2", "4", "6", "8")							#thread
app = ("scalar_unopt", "scalar_one", "scalar_two", "scalar_three", "simd_unopt", "simd_two", "p_scalar_unopt", "p_scalar_one", "p_scalar_two", "p_simd_unopt", "p_simd_two", "p_simd_twoD")				#kernel
bar_font = 12

src = ("l1", "l2", "i", "e")
n_col = (8, 8, 5, 4)
#n_row = (50, 100, 50, 100)
n_row = (N_ROWS, N_ROWS, N_ROWS, N_ROWS)

base_indx = 0
data = [[[[0 for x in xrange(N_FREQ)] for x in xrange(MAX_N_COL)] for x in xrange(N_THREADS)] for x in xrange(N_KERNL)]

exec_time = [[[[0 for x in xrange(N_FREQ)] for x in xrange(MAX_N_COL)] for x in xrange(N_THREADS)] for x in xrange(N_KERNL)]

colors=('darkcyan', 'goldenrod', 'g', 'b', 'r', 'maroon')
markers=('*', '.', 's', '*', '^', '*')
msize=(13, 10, 5, 11, 7, 11)
markColor=('darkcyan', 'goldenrod', 'g', 'b', 'r', 'maroon')
#colors=('coral', 'moccasin', 'ivory')
#markColor=('maroon', 'gold', 'ivory', 'goldenrod', 'darkcyan', 'g')
#colors=('lightgray', 'moccasin', 'ivory', 'goldenrod', 'slateblue')
hatches = ['','//','xx','..','o', '/']
width=0.18

p0={}
p1={}
p2={}
p3={}
p4={}
p5={}

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
	
	print "File name: %s" %(src)
	
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
				print line
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
	fd.writelines('%15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %25s %15s %15s\n' % ("#kernel_0", "thread_1", "freq_2", "exec_time_3", "l1_miss_rate_4", "l1_misses_5", "l1_accesses_6", "l1_D_prefetch_7", "l1_D_write_8", "tot_instr_9", "exec_time_10", "l2_miss_rate_11", "l3_miss_rate_12", "l2_misses_13", "l2_accesses_14", "l3_misses_15", "l3_accesses_16", "exec_time_17", "stalled_memsub_18", "stalled_resrc_19", "tot_cyc_20", "exec_time_21", "core_eng_22", "pkg_eng_23", "EDP_core_24", "speedup_25", "EDP_relative_26"))
	
	for a in xrange(N_KERNL):
		for t in xrange(N_THREADS):
			for f in xrange(N_FREQ):
				fd.writelines('%15s %15s %15d ' %(app[a], thr[t], freq[f]))
				for c in xrange(1, base_indx+1):
					fd.writelines('%15.2f ' %(res_set[a][t][c][f]))
				#fd.writelines('%15.2f ' %(res_set[a][t][base_indx][f] * res_set[a][t][base_indx-1][f]))
				fd.writelines('%25.2f ' %(res_set[a][t][1][f] * res_set[a][t][20][f]))
				fd.writelines('%15.2f ' %(res_set[0][0][1][f]/res_set[a][t][1][f]))
				#fd.writelines('%15.2f ' %( (res_set[0][0][base_indx][f] * res_set[0][0][base_indx-1][f])/(res_set[a][t][base_indx][f] * res_set[a][t][base_indx-1][f]) ))
				fd.writelines('%15.2f ' %( (res_set[0][0][1][f] * res_set[0][0][20][f])/(res_set[a][t][1][f] * res_set[a][t][20][f])))
				fd.writelines('\n')
	fd.close()


####################
# Draw Line Graphs
####################	
def draw_line_graph(data_file, col_no, plot_name, ylabel):

	data_set = [[[0 for x in xrange(N_FREQ)] for x in xrange(N_KERNL)] for x in xrange(N_THREADS)]

	with open(data_file, 'r') as fd:
		line = fd.readline()
		for a in xrange(N_KERNL):
			for t in xrange(N_THREADS):
				for f in xrange(N_FREQ):
					line = fd.readline()
					if not line: break
					values = line.split()
					#data_set[t][a][f] = (float(values[18])*100)/float(values[20])
					data_set[t][a][f] = float(values[col_no])
					
					if (col_no == CORE_EDP):
						data_set[t][a][f] = float(values[3])*float(values[22])
					
					
	fd.close()

	ind = np.arange(N_FREQ)

	for t in xrange(1): #N_THREADS
		fig = plt.figure()
		ax1 = fig.add_subplot(1,1,1)
		ax1.set_xticks(ind, freq)
		ax1.set_xticklabels(['', '800', '1200', '1600','2100','2500','2900','3500'], ha='left')
		
		
		for a in xrange(6): #N_KERNL
			p0[a] = ax1.plot(freq, data_set[t][a], '.--', color=colors[a], markersize=msize[a], lw=3, dashes=[4,2], marker=markers[a])
			y_start, y_end = ax1.get_ylim()
			if(col_no == EXEC_TIME):
				ax1.set_ylim(bottom=1000, top=1000000)
				ax1.set_yscale("log")
			elif(col_no == CORE_ENG):
				ax1.set_ylim(bottom=1, top=1000)
				ax1.set_yscale("log")				
			elif(col_no == CORE_EDP):
				ax1.set_ylim(bottom=320000, top=82850000)
				ax1.set_yscale("log")
			plt.ylabel(ylabel, fontsize=18)
			ax1.set_xlim(480, 4000)
#			print data_set[t][a]

		plt_name = ""
		plt_name = plot_name + "_" + str(thr[t]) + ".pdf"
		#plt.title('Execution time at different core frequency')
		l1=ax1.legend( (p0[0][0], p0[1][0], p0[2][0], p0[3][0], p0[4][0], p0[5][0]), ('Scalar_E0', 'Scalar_E1', 'Scalar_E2', 'Scalar_E3', 'SSE_E0', 'SSE_E2'), loc=1, ncol=3,shadow=False, prop={'size':16} )
		plt.savefig(plt_name, format='pdf')
	
#	plt.show()


def draw_broken_graph(data_file, col_no, plot_name, ylabel):

	data_set = [[[0 for x in xrange(N_FREQ)] for x in xrange(N_KERNL)] for x in xrange(N_THREADS)]

	with open(data_file, 'r') as fd:
		line = fd.readline()
		for a in xrange(N_KERNL):
			for t in xrange(N_THREADS):
				for f in xrange(N_FREQ):
					line = fd.readline()
					if not line: break
					values = line.split()
					#data_set[t][a][f] = (float(values[18])*100)/float(values[20])
					data_set[t][a][f] = float(values[col_no])
					
					if (col_no == CORE_EDP):
						data_set[t][a][f] = float(values[3])*float(values[22])
					
					
	fd.close()

	ind = np.arange(N_FREQ)

	fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)
#	ax1 = fig.add_subplot(1,1,1)		

	ax2.set_xticks(ind, freq)
	ax2.set_xticklabels(['', '800', '1200', '1600','2100','2500','2900','3500'], ha='left')

	for t in xrange(1): #N_THREADS
		
		p0[0] = ax1.plot(freq, data_set[t][0], '.--', color=colors[0], markersize=msize[0], lw=3, dashes=[4,2], marker=markers[0])
		p0[1] = ax1.plot(freq, data_set[t][1], '.--', color=colors[1], markersize=msize[1], lw=3, dashes=[4,2], marker=markers[1])
		p0[2] = ax1.plot(freq, data_set[t][2], '.--', color=colors[2], markersize=msize[2], lw=3, dashes=[4,2], marker=markers[2])
		p0[3] = ax1.plot(freq, data_set[t][3], '.--', color=colors[3], markersize=msize[3], lw=3, dashes=[4,2], marker=markers[3])
		p0[4] = ax2.plot(freq, data_set[t][4], '.--', color=colors[4], markersize=msize[4], lw=3, dashes=[4,2], marker=markers[4])
		p0[5] = ax2.plot(freq, data_set[t][5], '.--', color=colors[5], markersize=msize[5], lw=3, dashes=[4,2], marker=markers[5])
		y_start, y_end = ax1.get_ylim()
		
	ax1.set_ylim(bottom=500000, top=30000000)
	ax1.set_yscale("log")
	ax2.set_yscale("log")
	ax2.set_ylim(bottom=10000, top=100000)
		
#	plt.ylabel(ylabel, fontsize=18)
	fig.text(0.04, 0.5, ylabel, ha='center', va='center', rotation='vertical', fontsize=18)
	ax1.set_xlim(480, 4000)
#	print data_set[t][a]

	plt_name = ""
	plt_name = plot_name + "_" + str(thr[t]) + ".pdf"
	#plt.title('Execution time at different core frequency')
	l1=ax1.legend( (p0[0][0], p0[1][0], p0[2][0], p0[3][0], p0[4][0], p0[5][0]), ('Scalar_E0', 'Scalar_E1', 'Scalar_E2', 'Scalar_E3', 'SSE_E0', 'SSE_E2'), loc=1, ncol=3,shadow=False, prop={'size':16} )


	d = .015  # how big to make the diagonal lines in axes coordinates
	# arguments to pass plot, just so we don't keep repeating them
	kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
	ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
	ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

	kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
	ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
	ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
	
	# hide the spines between ax and ax2
	ax1.spines['bottom'].set_visible(False)
	ax2.spines['top'].set_visible(False)
	ax1.xaxis.tick_top()
	ax1.tick_params(labeltop='off')  # don't put tick labels at the top
	ax2.xaxis.tick_bottom()

#	plt.show()
	plt.savefig(plt_name, format='pdf')
	
#################################
# Create Bar-chart
#################################
def create_bar_chart(data_file, col_no, plot_name, ylabel):

	data_set = [[[0 for x in xrange(N_THREADS)] for x in xrange(N_KERNL)] for x in xrange(N_FREQ)]
	data_val = [[[0 for x in xrange(N_THREADS)] for x in xrange(N_KERNL)] for x in xrange(N_FREQ)]
	
	base_val = [0 for x in xrange(N_FREQ)]

	with open(data_file, 'r') as fd:
		line = fd.readline()
		for a in xrange(N_KERNL):
			for t in xrange(N_THREADS):
				for f in xrange(N_FREQ): #N_FREQ
					line = fd.readline()
					if not line: break
					values = line.split()					
#					print values
					if ( (a==0) and (t==0)):
						if(col_no == SPEEDUP):
							base_val[f] = float(values[3])
						elif(col_no == REL_EDP):
							base_val[f] = float(values[3])*float(values[22])
						print ("############################\n")
					if (col_no == SPEEDUP):
						if(a > 5):
							data_set[f][a-6][t] = base_val[f]/float(values[3])
							print base_val[f], float(values[3])
					elif (col_no == REL_EDP):
						if(a > 5):
							data_set[f][a-6][t] = (base_val[f])/(float(values[3])*float(values[22]))
					elif (col_no == MEM_STALL):
						if(a > 5):
							data_set[f][a-6][t] = (float(values[18]) * 100 )/(float(values[20]))
	fd.close()
	ind = np.arange(N_FREQ)
	
	print('****************************************\n')
	ind = np.arange(N_THREADS)

	for j in xrange(N_FREQ): #N_FREQ
	
		fig = plt.figure()
		ax1 = fig.add_subplot(1,1,1)
		print ('\nFreq %d ' %j)
		for i in xrange(5):#N_KERNL
			print(data_set[j][i])
			if (i==4):
				p1[i] = ax1.bar(ind+i*width, data_set[j][i+1], width, color=colors[i+1], hatch=hatches[i+1])
			else:
				p1[i] = ax1.bar(ind+i*width, data_set[j][i], width, color=colors[i], hatch=hatches[i])

		for i in xrange(N_THREADS):	
			ax1.text(i, 		data_set[j][0][i]+.1, ' %.1f'%data_set[j][0][i], va='bottom', fontsize=bar_font, ha='left', rotation=90)	
			ax1.text(i+width, 	data_set[j][1][i]+.1, ' %.1f'%data_set[j][1][i], va='bottom', fontsize=bar_font, ha='left', rotation=90)	
			ax1.text(i+2*width, data_set[j][2][i]+.1, ' %.1f'%data_set[j][2][i], va='bottom', fontsize=bar_font, rotation=90)	
			ax1.text(i+3*width, data_set[j][3][i]+.1, ' %.1f'%data_set[j][3][i], va='bottom', fontsize=bar_font, rotation=90)	
			ax1.text(i+4*width, data_set[j][5][i]+.1, ' %.1f'%data_set[j][5][i], va='bottom', fontsize=bar_font, rotation=90)	

		ax1.set_xticks(ind-width, freq)
		ax1.set_xticklabels(['0', '1', '2','4','6','8'], ha='left')

		y_start, y_end = ax1.get_ylim()
		if (col_no == SPEEDUP):
			ax1.set_ylim(bottom=0, top=35)
		elif (col_no == REL_EDP):
			ax1.set_ylim(bottom=0, top=500)
		elif (col_no == MEM_STALL):
			ax1.set_ylim(bottom=0, top=30)
			
		ax1.set_xlim(-.2, 5.1)

		ax1.set_ylabel(ylabel, fontsize=15)
		ax1.set_xlabel('No of threads', fontsize=15)
		l1=ax1.legend( (p1[0], p1[1], p1[2], p1[3], p1[4]), ('Scalar_E0', 'Scalar_E1', 'Scalar_E2', 'SSE_E0', 'SSE_E2'), loc=2, ncol=3,shadow=False, prop={'size':16} )

		plt_name = ""
		plt_name = plot_name + "_" + str(freq[j]) + ".pdf"

		plt.savefig(plt_name, format='pdf')	
#		plt.show()

	
def mains():

	output_file = "jlpea_parallel.txt"
	res = [[[[0 for x in xrange(N_FREQ)] for x in xrange(MAX_N_COL)] for x in xrange(N_THREADS)] for x in xrange(N_KERNL)]
	
	# Remove outliers from the raw dataset
#	for a in xrange (N_RESULT_SET):
#		remove_outliers(src[a], res, n_row[a], n_col[a])
		
	# Generate mean results 
#	summarize_results(res, output_file)
	
#	draw_line_graph(output_file, EXEC_TIME, "exec_time", "Execution time ($m s$)")
#	create_bar_chart(output_file, SPEEDUP, "speedup", "Speedup")
#	draw_line_graph(output_file, CORE_ENG, "core_eng", "Core energy consumption ($mj$)")	
#	draw_broken_graph(output_file, CORE_EDP, "core_edp", "Core energy efficiency (EDP)")
	create_bar_chart(output_file, REL_EDP, "rel_edp", "Relative core energy efficiency (EDP)")
#	create_bar_chart(output_file, MEM_STALL, "mem_stall", "Stalled cycles in memory subsystems(%)")
	
	#draw_line_graph(output_file, INSTR_COUNT, "instruction", "Total number of instructions")
	#draw_line_graph(output_file, 11, "core_edp", "Core energy efficiency (EDP)")
	
mains()
