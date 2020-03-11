if [[ $# -lt 2 ]]
then
    echo "Usage: $0 KATAEXEC BASEDIR"
    echo "KATAEXEC path to the KataGo executable"
    echo "BASEDIR containing selfplay data and models and related directories"
    exit 0
fi

KATAEXEC="$1"
shift
BASEDIR="$1"
shift


"$KATAEXEC" gatekeeper -rejected-models-dir "$BASEDIR"/rejectedmodels -accepted-models-dir "$BASEDIR"/models/ -sgf-output-dir "$BASEDIR"/gatekeepersgf/ -test-models-dir "$BASEDIR"/modelstobetested/ -config-file keeper1.cfg -quit-if-no-nets-to-test
