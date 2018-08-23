#!/bin/sh
# Adds crt certificates from a single file to Java keystore

set -x

cacerts=$1
if ! test -f "$cacerts" ; then
  echo "Usage: $0 CERTS_FILE"
  exit 1
fi
if ! test -d "$JAVA_HOME" ; then
  echo "JAVA_HOME is not set. It should be like '/usr/lib/jvm/java-8-openjdk-amd64/jre'"
  exit 1
fi
keytool=`which keytool`
keystore=$JAVA_HOME/lib/security/cacerts
pems_dir=/tmp/pems
rm -rf "$pems_dir" 2>/dev/null || true
mkdir "$pems_dir"
cd "$pems_dir"
awk 'BEGIN {c=0;doPrint=0;} /END CERT/ {print > "cert." c ".pem";doPrint=0;} /BEGIN CERT/{c++;doPrint=1;} { if(doPrint == 1) {print > "cert." c ".pem"} }' < $cacerts
for f in `ls cert.*.pem`; do
  alias=`basename $f`
  $keytool -import -trustcacerts -noprompt -keystore "$keystore" -alias "$alias" -file "$f" -storepass changeit;
done
