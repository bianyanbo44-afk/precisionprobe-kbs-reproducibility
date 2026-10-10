"""Rejudge retained public observations for Amendment 07 without execution."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import ast, hashlib, json, re
from statement_comparators import FUNCTIONAL_TASK_POLICIES, matches_statement

BASE = Path(__file__).resolve().parents[1]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]

def type_name(value):
    if isinstance(value, list):
        return 'list[' + ','.join(sorted({type_name(x) for x in value})) + ']'
    if isinstance(value, dict):
        return 'dict'
    return type(value).__name__

def main():
    questions = {q['task_id']:q for q in rows(BASE/'data/livecodebench/questions.jsonl')}
    tests = {q['task_id']:q for q in rows(BASE/'data/livecodebench/tests.jsonl')}
    run=BASE/'runs/livecodebench_v6_increment'
    previous=json.loads((run/'statement_correction_receipt.json').read_text())
    assert digest(BASE/'scripts/statement_comparators_v2.py') == previous['comparator_script_sha256']
    inventory=[]
    for task,q in questions.items():
        if not q['metadata'].get('func_name'):
            continue
        output_types=Counter(type_name(json.loads(case['output'])) for case in tests[task]['public']+tests[task]['private'])
        tolerance=bool(re.search(r'Answers within',q['question_content'],re.I))
        alternatives=bool(re.search(r'(?:return|output).{0,50}(?:any order|any of them|any valid|any answer)|multiple (?:valid )?answers',q['question_content'],re.I))
        inventory.append({'task_id':task,'title':q['question_title'],'function':q['metadata']['func_name'],
                          'expected_json_types':dict(output_types),'explicit_numeric_tolerance':tolerance,
                          'explicit_alternative_output_clause':alternatives,
                          'rule':'finite numeric absolute tolerance 1e-5' if task in FUNCTIONAL_TASK_POLICIES else 'literal/JSON equality'})
    assert len(inventory)==63
    assert {x['task_id'] for x in inventory if x['explicit_numeric_tolerance']}==set(FUNCTIONAL_TASK_POLICIES)
    inventory_path=BASE/'analysis/validation/functional_oracle_inventory.json'
    inventory_path.write_text(json.dumps({'scope':'Pinned statements and expected JSON types for 63 functional tasks; no private test payloads included.',
                                          'questions_sha256':digest(BASE/'data/livecodebench/questions.jsonl'),
                                          'tests_sha256':digest(BASE/'data/livecodebench/tests.jsonl'),'tasks':inventory},indent=2),encoding='utf-8')
    details=[]
    config=json.loads((BASE/'protocol/validation_config.json').read_text())
    for spec in config['models']:
        path=run/'evaluations'/(spec['condition']+'.jsonl')
        record=next(r for r in rows(path) if r['task_id']=='leetcode/3763')
        public=tests['leetcode/3763']['public']
        observations=record['observations'][0]
        assert len(public)==len(observations)==2
        corrected=[matches_statement(obs,case['output'],case['testtype'],record['task_id'],case['input']) for obs,case in zip(observations,public)]
        if all(corrected):
            raise RuntimeError('A target became publicly passing; full sandbox private evaluation is required before applying this amendment')
        assert corrected==record['public_test_passes']
        assert record['error']==1 and record['private_tests_checked']==0
        details.append({'condition':spec['condition'],'task_id':record['task_id'],'solution_sha256':record['solution_sha256'],
                        'evaluation_sha256':digest(path),'before_public_passes':record['public_test_passes'],
                        'after_public_passes':corrected,'before_error':record['error'],'after_error':1,
                        'private_tests_required_after_correction':False,'private_tests_executed':0,
                        'observations_reused':True,'evaluation_record_unchanged':True})
    receipt={'version':'statement-aware-v3','created_at_utc':datetime.now(timezone.utc).isoformat(),'applied':True,
             'method':'Comparator-only rejudgement of retained public observations; no program or model execution.',
             'previous_statement_receipt_sha256':digest(run/'statement_correction_receipt.json'),
             'previous_comparator_script_sha256':digest(BASE/'scripts/statement_comparators_v2.py'),
             'comparator_script_sha256':digest(BASE/'scripts/statement_comparators.py'),
             'rejudgement_script_sha256':digest(__file__),'functional_inventory_sha256':digest(inventory_path),
             'amendment_07_sha256':digest(BASE/'protocol/amendment_07_functional_tolerance_and_decision_reporting.md'),
             'initial_lock_sha256':digest(BASE/'protocol/validation_lock.json'),'policies':FUNCTIONAL_TASK_POLICIES,
             'assessed_condition_task_pairs':len(details),'changed_error_labels':0,'changed_public_failure_statuses':0,
             'unchanged_generation_score_inputs_and_acquisition_times':True,'evaluation_files_unchanged':True,'details':details}
    (run/'functional_correction_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps({'version':receipt['version'],'functional_tasks_inventoried':len(inventory),
                      'numeric_types':[x['task_id'] for x in inventory if any('float' in t for t in x['expected_json_types'])],
                      'alternative_output_clauses':[x['task_id'] for x in inventory if x['explicit_alternative_output_clause']],
                      'assessed_targets':len(details),'changed_error_labels':0,'changed_public_failure_statuses':0},indent=2))

if __name__=='__main__':
    main()
