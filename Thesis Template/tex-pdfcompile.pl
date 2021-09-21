#!/usr/bin/perl
# This file compiles a LaTeX  document
# into PDF format using pdflatex
# Pedro Brandão
use strict;
use POSIX;
use Cwd;
use Getopt::Long;

# Globals
# #######
my $FILE;
my $CDIR= getcwd();
if ( $^O =~ /WIN/i)
{
   $CDIR = "$CDIR\\";
}
else
{
   $CDIR = "$CDIR/";
}

my $PDFLATEXOPTIONS = " -file-line-error --halt-on-error --synctex=-1 -src-specials -interaction=nonstopmode";
my $TEXEXT ="tex";
my $IDXEXT ="idx";

my $nrOfRuns = 3;
my $logToFileOpt = undef;
my $makeIndex = 0;
my $cleanTex = 0;
my $result = GetOptions ("runs|r=i" => \$nrOfRuns, # int
            "logfile|l" =>\$logToFileOpt, #string
            "clean|d" =>\$cleanTex, #string
            "makeindex|mi|m"=>\$makeIndex
            ); 

usage() if ( !@ARGV );



# ----------------------------------------------------------------------
if($cleanTex) {
   log_ ("Cleaning tex auxiliary files  ... ");
   system("texclean.pl")==0 || fatal ("texclean.pl run failed.");
}
# ----------------------------------------------------------------------
#
my $FILE=shift;

$FILE=~s/\..+$//;

if(! -e $FILE) {
   $FILE = "$FILE.$TEXEXT";
   fatal("$FILE (with and without extension) does not exist!") if( ! -e $FILE);
}

my $TEMPDIR="/tmp/";
$TEMPDIR = 'c:\\TEMP\\' if ( $^O =~ /WIN/i);

my $LOGFILE = undef;
if(defined($logToFileOpt)){ # defined
   $LOGFILE="$TEMPDIR$FILE.$$.out";
}
my $logOutput = "";
if($LOGFILE) {
   $logOutput = ">> $LOGFILE 2>&1";
}
my ($BASEFILE)= $FILE=~/(.*)\.$TEXEXT$/;


#print "CDIR [$CDIR]; LOGFILE [$LOGFILE]; BASEFILE [$BASEFILE]; FILE [$FILE]"
my $DOBIBLIO = 0;

open(FD,"$FILE") or fatal("Error opening $FILE");
$DOBIBLIO = 1 if( grep(/^[^%]*\\bibliography\{.+\}/, <FD>));
close(FD);

log_ ("Start ($FILE): ". strftime("%Y/%m/%d-%H:%M",localtime));
# ----------------------------------------------------------------------
log_ ("Running latex (run 1 of 3) ... ");
system("pdflatex $PDFLATEXOPTIONS $FILE $logOutput")==0 || fatal ("latex run 1 failed.");
doneOK() if($nrOfRuns <2);
if ($DOBIBLIO == 1)
{
    log_ ("Running bibtex ... ");
    system("bibtex $BASEFILE $logOutput") ==0   || fatal( "bibtex failed."); 
}
# ----------------------------------------------------------------------
log_ ("Running latex (run 2 of 3) ... ");
system("pdflatex $PDFLATEXOPTIONS $FILE $logOutput")==0 || fatal ("latex run 2 failed.");
doneOK() if($nrOfRuns <3);
# ----------------------------------------------------------------------
if($makeIndex && -e "$BASEFILE.$IDXEXT") {
   log_ ("Running makeindex ... ");
   system("makeindex $BASEFILE $logOutput")==0 || fatal ("makeindex run failed.");
}
# ----------------------------------------------------------------------
log_ ("Running latex (run 3 of 3) ... ");
system("pdflatex $PDFLATEXOPTIONS $FILE $logOutput")==0 || fatal ("latex run 3 failed.");
# ----------------------------------------------------------------------
log_ ("Running latex (run 4 due to index use) ... ");
system("pdflatex $PDFLATEXOPTIONS $FILE $logOutput")==0 || fatal ("latex run 4 failed.");

# ----------------------------------------------------------------------
doneOK();


#---------------------------------------------------------------------
# Aux funcs
sub doneOK() {
   print "All done.\n";
   if(defined($LOGFILE)) {
      rename($LOGFILE,"$CDIR$FILE.out");
      print "\t See $CDIR$FILE.out for details.";
   }
   exit(0);
}


sub usage {
	print "Usage: $0 <file> [LOGFILE]\n";
	print "\tWhere <file> is the name or path of a LaTeX file.\n";
    print "\t\tand if [LOGFILE] is defined the output will be logged to a file.\n\t\tNote that by default we don't do it.";
	exit 0;
}

sub fatal {
        my $msg=shift;
	     print "ERROR: $msg. ";
        if($LOGFILE) {
           print "See $FILE.failed for details.\n";
           rename($LOGFILE, "$CDIR$FILE.failed");
        }
        exit 1;
}

sub ok {
	my $msg=shift;
	print "INFO: $msg\n";
}

sub log_ {
   my $msg = shift;
	print "INFO: $msg\n";
   my $size = 50;
   my $space1 = ($size - length($msg) -2)/2;
   my $space2=$space1;
   $space2-- if(length($msg)%2);
   if($LOGFILE) {
      open(LOGFD,">>$LOGFILE") or die "Unable to open $LOGFILE for writing:$!\n";
      print LOGFD "#"x$size, "\n";
      print LOGFD "#", " "x$space1 . $msg . " "x$space2 . "#", "\n";
      print LOGFD "#"x$size, "\n";
      close(LOGFD);
   }
}
#---------------------------------------------------------------------
