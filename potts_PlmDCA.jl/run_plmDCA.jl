

using Printf
using PlmDCA
using MAT
function comp(a, b)
	if a[1] < b[1]
		return true
	end
	if a[1] > b[1]
		return false
	end
	
	if a[2] < b[2]
		return true
	end
	if a[2] > b[2]
		return false
	end
	return false
end

function main()
	inputfile = ARGS[1]
	outputfile = ARGS[2]
	res=plmdca(inputfile)
	
	file = matopen(string(outputfile,".mat"), "w")
	write(file, "J",res.Jtensor)
	write(file, "h",res.htensor)
	llen=size(res.htensor)[2]
	score_array=zeros(llen,llen)
	
	for xx in res.score
		score_array[xx[1],xx[2]] = xx[3]
		score_array[xx[2],xx[1]] = xx[3]
	end
	write(file, "score",score_array)
	
	write(file, "frobenius_norm",PlmDCA.compute_APC(res.Jtensor,size(res.htensor)[2],size(res.Jtensor)[1]))
	
	close(file)
	
	
	sort!(res.score, lt= comp)
	open(outputfile,"w") do out
		for rr in res.score
			rres = Printf.@sprintf("%d,%d,%f",rr[1],rr[2],rr[3])
			println(out,rres)
		end
	end
end

main()

#    def potts(s):
#        filename = s.base + ".mat"
#        potts = loadmat(filename)
#        s.data['J'] = potts['J'].astype('float16')
#        s.data['h'] = potts['h']
#        s.data['frobenius_norm'] = potts['frobenius_norm']
#        s.data['score'] = potts['score']

    
