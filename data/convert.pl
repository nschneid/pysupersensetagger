#!/usr/bin/env perl

#script to convert semcor files from Ciaramita and Altun's C++ 
#supersense tagger distribution (SEM.BI) to the format readable by
#the java SS tagger.

while(<>){
	@F = split(/\t/);
	
	for($i=1; $i<@F; $i++){
		@P = split(/\s+/, $F[$i]);
		print "$P[0]\t$P[1]\t$P[2]\t$F[0]\n";
	}	

	print "\n";
}