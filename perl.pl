#!/usr/bin/perl


use strict;
use warnings;
use English;
use Data::Dumper qw(Dumper);
 
print Dumper \@ARGV;

my $dir = $ARGV[0];
my $FILEOUT = $ARGV[1]||"ML.txt";



##system "curl '$DIR' > '$DIR'" ;

foreach my $fp (glob("$dir/*.txt")) {
  #printf "%s\n", $fp;
  open my $fh, "<", $fp or die "can't read open '$fp': $OS_ERROR";
  	##$row =~ /^(.*)\t(.*)$/;

  	while(<$fh>){
  		my @names;
  		push @names, $_;
	    if($_=~ /(Opcion escogida:.*)/ || $_=~/(Classification accuracy.*)/)
	    {
	        #print $_;
	        my$entry=<$fh>;
	        push (@names, $entry);
	        #print $entry;

	    }
	    else
	    {
	        next;
	    }
	}
	print "$name[1]\n";

  close $fh or die "can't read close '$fp': $OS_ERROR";
}
