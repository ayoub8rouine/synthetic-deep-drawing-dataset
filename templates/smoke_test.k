*KEYWORD
*TITLE
LS-DYNA Student command-line smoke test - one elastic shell
*CONTROL_TERMINATION
$ endtim    endcyc     dtmin    endeng    endmas
1.000E-06,0,0.0,0.0,0.0
*DATABASE_GLSTAT
1.000E-06
*PART
smoke_test_shell
$ pid,secid,mid,eosid,hgid,grav,adpopt,tmid
1,1,1,0,0,0,0,0
*SECTION_SHELL
$    secid    elform      shrf       nip     propt   qr/irid     icomp
1,2,0.833,2,1,0,0
$       t1        t2        t3        t4
1.0,1.0,1.0,1.0
*MAT_ELASTIC
$      mid        ro         e        pr
1,7.850E-09,210000.0,0.30
*NODE
$      nid               x               y               z
1,0.0,0.0,0.0
2,10.0,0.0,0.0
3,10.0,10.0,0.0
4,0.0,10.0,0.0
*ELEMENT_SHELL
$      eid       pid        n1        n2        n3        n4
1,1,1,2,3,4
*END
