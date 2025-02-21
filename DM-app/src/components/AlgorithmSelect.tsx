import { useState } from "react";
import { Option } from "./types";
import { FormField, Select, SelectProps } from "@cloudscape-design/components";
import { ALGORITHM_OPTIONS } from "../consts/algorithmConsts";

interface  AlgorithmSelectProps {
    label: string;
    selectedAlg: Option | null;
    handleSelection: SelectProps['onChange'];
}

export const AlgorithmSelect: React.FC<AlgorithmSelectProps> = ({label, selectedAlg, handleSelection}) => {
    return (
        <FormField label={label}>
            <Select
                selectedOption={selectedAlg}
                options={ALGORITHM_OPTIONS}
                placeholder="Select ranking algorithm ..."
                onChange={handleSelection}
            />
        </FormField>
    );
}

export const useAlgorithmSelection = () => {
    const [algorithm, setAlgorithm] = useState<Option | null>(null);

    const handleAlgorithmSelect: SelectProps['onChange'] = (e) => {
        setAlgorithm(e.detail.selectedOption);
    };

    return {
        algorithm,
        handleAlgorithmSelect
    }
};